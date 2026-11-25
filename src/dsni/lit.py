"""Lightning modules and the command-line entry point for the neural arms.

``_LitBase`` owns the steps, the masked multi-task loss, the metrics and the optimiser. Each
arm subclasses it with ``__init__`` and ``forward`` only; the defaults of those ``__init__``
signatures are the paper's settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from jsonargparse import lazy_instance
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dsni.data import DSNIDataModule, load_splits
from dsni.metrics import evaluate, reusable, write_result
from dsni.models import CNNEncoder, MultiTaskHead, TransformerEncoder
from dsni.schema import CLASSES, IGNORE_INDEX, LOSS_WEIGHTS, TASKS

MONITOR = "val/joint_acc"
BUDGET = (  # trainer settings that change what a run learns
    "max_epochs",
    "min_epochs",
    "max_steps",
    "max_time",
    "limit_train_batches",
    "limit_val_batches",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "precision",
)
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # tokens, labels, crop offsets


def masked_loss(logits: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Weighted sum of the per-task cross-entropies over the rows that have each label."""
    total = labels.new_zeros((), dtype=torch.float)
    for i, task in enumerate(TASKS):
        if (labels[:, i] != IGNORE_INDEX).any():  # an all-missing batch would give NaN
            loss = F.cross_entropy(logits[task], labels[:, i], ignore_index=IGNORE_INDEX)
            total = total + LOSS_WEIGHTS[task] * loss
    return total


class _LitBase(L.LightningModule):
    """Shared training logic; subclasses set ``self.encoder`` and define ``forward``."""

    def __init__(
        self,
        d_model: int,
        n_media: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = {"media": n_media, **{t: len(c) for t, c in CLASSES.items()}}
        self.head = MultiTaskHead(d_model, self.n_classes, dropout)
        self._outputs: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []

    def setup(self, stage: str) -> None:
        """Fail early when the heads do not match the label space of the data."""
        data = getattr(self.trainer, "datamodule", None)
        if data is not None and data.n_classes != self.n_classes:
            raise ValueError(f"model heads {self.n_classes} do not match data {data.n_classes}")

    def training_step(self, batch: Batch, batch_idx: int) -> Any:
        tokens, labels, offsets = batch
        loss = masked_loss(self(tokens, offsets), labels)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(tokens))
        return loss

    def _evaluate_batch(self, batch: Batch, stage: str) -> None:
        tokens, labels, offsets = batch
        logits = self(tokens, offsets)
        self.log(f"{stage}/loss", masked_loss(logits, labels), batch_size=len(tokens))
        self._outputs.append(({t: v.softmax(-1).cpu() for t, v in logits.items()}, labels.cpu()))

    def _evaluate_epoch(self, stage: str) -> None:
        # Predictions of the whole split are gathered on one device, so the metrics are the
        # same functions the evaluation notebook uses; this assumes single-device evaluation.
        proba = {t: torch.cat([p[t] for p, _ in self._outputs]).numpy() for t in TASKS}
        labels = torch.cat([y for _, y in self._outputs]).numpy()
        self._outputs.clear()
        self.log_dict({f"{stage}/{k}": v for k, v in evaluate(proba, labels).items()})

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        self._evaluate_batch(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._evaluate_epoch("val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        self._evaluate_batch(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._evaluate_epoch("test")

    def predict_step(self, batch: Batch, batch_idx: int) -> Any:
        tokens, _, offsets = batch
        return {task: logits.softmax(-1) for task, logits in self(tokens, offsets).items()}

    def configure_optimizers(self) -> Any:
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
        )
        total = int(self.trainer.estimated_stepping_batches)
        warmup = max(1, int(hp["warmup_ratio"] * total))
        schedule = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max(1, total - warmup)),
            ],
            milestones=[warmup],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}


class LitCNN(_LitBase):
    """The paper's ``CNN Encoder`` arm."""

    def __init__(
        self,
        d_model: int = 256,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        kernels: tuple[int, ...] = (7, 5, 3, 3),
        dropout: float = 0.1,
        n_media: int = 42,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
    ) -> None:
        """See :class:`dsni.models.CNNEncoder` for the architecture arguments."""
        super().__init__(d_model, n_media, lr, weight_decay, warmup_ratio, dropout)
        self.encoder = CNNEncoder(d_model, channels, kernels, dropout)

    def forward(
        self, tokens: torch.Tensor, offsets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Logits of every task."""
        return self.head(self.encoder(tokens, offsets))


class LitTransformer(_LitBase):
    """``Transformer (Flat)`` with ``gate="none"``, ``Transformer + Gating`` with ``gate="bio"``."""

    def __init__(
        self,
        gate: str = "none",
        gate_features: tuple[str, ...] = ("region", "position", "gc", "entropy"),
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        gate_dropout: float = 0.05,
        n_media: int = 42,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
    ) -> None:
        """See :class:`dsni.models.TransformerEncoder` for the architecture arguments."""
        super().__init__(d_model, n_media, lr, weight_decay, warmup_ratio, dropout)
        self.encoder = TransformerEncoder(
            d_model, n_layers, n_heads, d_ff, dropout, gate, gate_features, gate_dropout
        )

    def forward(
        self, tokens: torch.Tensor, offsets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Logits of every task."""
        return self.head(self.encoder(tokens, offsets))


class DSNICLI(LightningCLI):
    """LightningCLI with ``--run_dir out/<version>/<arm>/s<seed>`` naming the run."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add ``--run_dir`` and tie the log directory and the data seed to it."""
        parser.add_argument("--run_dir", type=str, default="out/dev/model/s0")
        parser.link_arguments("run_dir", "trainer.default_root_dir")
        parser.link_arguments("run_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("seed_everything", "data.seed")


def cli_main(args: list[str] | None = None) -> Path:
    """Train one arm and write its result; a finished run with the same recipe is not repeated.

    Args:
        args: Command-line options, for example ``["--model", "LitTransformer"]``;
            ``None`` reads ``sys.argv``.

    Returns:
        The path of ``result.pt``: the best checkpoint by validation joint accuracy is tested
        (logged to the CSV log) and its val and test predictions are scored and saved.
    """
    cli = DSNICLI(
        _LitBase,
        DSNIDataModule,
        subclass_mode_model=True,
        run=False,
        seed_everything_default=0,
        trainer_defaults={
            "max_epochs": 100,
            "devices": 1,  # predictions are gathered on one device; see _evaluate_epoch
            "num_sanity_val_steps": 0,
            "callbacks": [
                ModelCheckpoint(monitor=MONITOR, mode="max", filename="best"),
                EarlyStopping(monitor=MONITOR, mode="max", patience=15),
            ],
            "logger": lazy_instance(CSVLogger, save_dir="out", name="", version="logs"),
        },
        save_config_kwargs={"overwrite": True},
        args=args,
    )
    trainer, model, data = cli.trainer, cli.model, cli.datamodule
    run_dir, seed = Path(trainer.default_root_dir), cli.config.seed_everything
    # What must match for an existing result to be reused; file locations, logging and the
    # hardware are left out, the training budget is not.
    recipe = {k: v for k, v in data.hparams.items() if k not in ("path", "num_workers")}
    settings = cli.config.trainer
    budget = {key: settings.get(key) for key in BUDGET}
    config = {"model": dict(model.hparams), "data": recipe, "trainer": budget}
    tables = load_splits(*(data.hparams[k] for k in ("path", "split", "train_fraction", "seed")))
    if done := reusable(run_dir, seed, config, tables):
        return done

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best", verbose=False)  # leaves the best weights loaded
    proba = {split: predict_proba(model, data.loader(split)) for split in ("val", "test")}
    config["checkpoint"] = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
    return write_result(run_dir, seed, config, data.tables, proba)


@torch.no_grad()
def predict_proba(model: L.LightningModule, loader: DataLoader) -> dict[str, np.ndarray]:
    """Class probabilities of every task for every row of ``loader``, in loader order."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    parts: dict[str, list[torch.Tensor]] = {task: [] for task in TASKS}
    for tokens, _, offsets in loader:
        for task, logits in model(tokens.to(device), offsets.to(device)).items():
            parts[task].append(logits.softmax(-1).cpu())
    model.train(was_training)
    return {task: torch.cat(chunks).numpy() for task, chunks in parts.items()}


def export_checkpoint(source: str | Path, dest: str | Path, **provenance: Any) -> Path:
    """Copy a Lightning checkpoint without its optimiser and loop state, for shipping.

    The result loads with ``LitTransformer.load_from_checkpoint(dest)`` as usual; the keyword
    arguments are stored under ``"dsni"`` to record how the weights were produced.
    """
    full = torch.load(source, map_location="cpu", weights_only=False)
    keep = ("state_dict", "hyper_parameters", "pytorch-lightning_version")
    slim = {key: full[key] for key in keep if key in full} | {"dsni": provenance}
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dest)
    return Path(dest)


def fit(run_dir: str | Path, **options: Any) -> Path:
    """Call :func:`cli_main` from Python.

    ``options`` are command-line options without the leading dashes, for example
    ``fit("out/demo/cnn/s0", model="LitCNN", **{"trainer.max_epochs": 5})``.
    """
    flags = [
        f"--{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in options.items()
    ]
    return cli_main([*flags, f"--run_dir={run_dir}"])
