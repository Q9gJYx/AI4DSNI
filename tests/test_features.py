import torch

from dsni.data import encode
from dsni.features import (
    GATE_FEATURES,
    V_REGIONS,
    local_entropy,
    local_gc,
    region_tags,
    relative_position,
)


def test_every_feature_has_the_shape_of_the_tokens():
    tokens = encode(["ACGT" * 100, "GGCC" * 50], max_len=512)
    for name, feature in GATE_FEATURES.items():
        values = feature(tokens, None)
        assert values.shape == tokens.shape, name
        assert torch.isfinite(values).all() and values.min() >= 0 and values.max() <= 1, name


def test_region_tags_follow_the_coordinates():
    tags = region_tags(torch.ones(2, 1500, dtype=torch.long))
    start, end = V_REGIONS[0]
    assert tags[0, start:end].all() and not tags[0, :start].any() and not tags[0, end]
    assert int(tags[0].sum()) == sum(e - s for s, e in V_REGIONS)


def test_region_tags_shift_with_the_crop_offset():
    tokens = torch.ones(2, 440, dtype=torch.long)
    full = region_tags(torch.ones(1, 1500, dtype=torch.long))[0]
    cropped = region_tags(tokens, torch.tensor([0, 500]))
    assert torch.equal(cropped[0], full[:440])
    assert torch.equal(cropped[1], full[500:940])


def test_relative_position_ends_at_one_on_the_last_base():
    tokens = encode(["ACGTACGTAC"], max_len=16)
    position = relative_position(tokens)
    assert position[0, 0] == 0 and position[0, 9] == 1 and position[0, 15] == 1


def test_gc_and_entropy_on_known_sequences():
    tokens = encode(["G" * 60, "A" * 60, "ACGT" * 15, "GC" * 10], max_len=60)
    gc, entropy = local_gc(tokens), local_entropy(tokens)
    assert torch.allclose(gc[0], torch.ones(60)) and torch.allclose(gc[1], torch.zeros(60))
    assert torch.allclose(entropy[0], torch.zeros(60), atol=1e-6)
    assert entropy[2, 30] > 0.99
    assert gc[3, 5] == 1 and gc[3, 45] == 0  # padding contributes no bases
