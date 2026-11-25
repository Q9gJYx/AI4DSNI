import pytest
import torch

from dsni.data import encode
from dsni.features import GATE_FEATURES
from dsni.models import BioGate, CNNEncoder, MultiTaskHead, TransformerEncoder, gate_profile

SEQUENCES = ["ACGTTGCA" * 20, "GGGCCCAT" * 12, "ACGT" * 30]
N_CLASSES = {"media": 42, "temperature": 3, "ph": 3, "oxygen": 4}


def transformer(gate, features=tuple(GATE_FEATURES)):
    return TransformerEncoder(32, 1, 2, 64, 0.1, gate, features, 0.05)


@pytest.mark.parametrize(
    "encoder",
    [CNNEncoder(32, (16, 32), (5, 3), 0.1), *map(transformer, ("none", "bio", "learned"))],
    ids=["cnn", "flat", "gated", "learned"],
)
def test_encoders_ignore_padding(encoder):
    encoder.eval()
    short, long = encode(SEQUENCES, max_len=160), encode(SEQUENCES, max_len=256)
    with torch.no_grad():
        a, b = encoder(short), encoder(long)
    assert a.shape == (3, 32)
    assert torch.allclose(a, b, atol=1e-5)


def test_only_the_gated_encoder_depends_on_the_crop_offset():
    tokens, offsets = encode(SEQUENCES, max_len=160), torch.tensor([0, 500, 900])
    flat, gated = transformer("none").eval(), transformer("bio").eval()
    with torch.no_grad():
        assert torch.equal(flat(tokens), flat(tokens, offsets))
        assert not torch.allclose(gated(tokens), gated(tokens, offsets))


def test_bio_gate_parameter_count_and_initialisation_match_the_paper():
    gate = BioGate(256, tuple(GATE_FEATURES), 0.05)
    assert sum(p.numel() for p in gate.parameters()) == 1280
    assert torch.equal(gate.linear.bias, torch.zeros(256))
    gate.eval()
    values = gate(encode(SEQUENCES, max_len=160))
    assert (values - 0.5).abs().max() < 0.05  # every gate starts near sigmoid(0)
    assert BioGate(256, ("position", "gc", "entropy"), 0.05).linear.in_features == 3


def test_gate_changes_the_parameter_count_by_the_gate_only():
    size = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert size(transformer("bio")) - size(transformer("none")) == 5 * 32
    with pytest.raises(ValueError, match="gate must be"):
        transformer("other")


def test_head_returns_logits_for_every_task():
    logits = MultiTaskHead(32, N_CLASSES, 0.1)(torch.randn(5, 32))
    assert {k: v.shape[1] for k, v in logits.items()} == N_CLASSES


def test_gate_profile_averages_over_sequences_and_skips_padding():
    encoder = transformer("bio")
    tokens = encode(["ACGT" * 40, "GGCC" * 10], max_len=160)
    profile = gate_profile(encoder, tokens)
    assert profile.shape == (160,) and torch.isfinite(profile).all()
    single = gate_profile(encoder, tokens[:1])
    assert torch.allclose(profile[40:], single[40:])  # only the first sequence reaches past 40
    assert encoder.training  # the mode is restored
    with pytest.raises(ValueError, match="no gate"):
        gate_profile(transformer("none"), tokens)
