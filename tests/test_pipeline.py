"""Fast tests that need no datasets or GPU (run with `pytest`)."""
import numpy as np
import pandas as pd
import pytest
import torch

from musicsim import config as C
from musicsim.audio import audio_to_multispec, crop_or_pad
from musicsim.gradcam import SiameseGradCAM
from musicsim.mcic import normalize_filename, score_pair, windows
from musicsim.melodysim import parse_relative_path, read_shard_list
from musicsim.model import SiameseNet
from musicsim.pairs import FixedPairs, TrainPairs


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SiameseNet().eval()


def test_multispec_shape_and_range():
    sr = 44_100
    t = np.arange(int(9.2 * sr)) / sr
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    m = audio_to_multispec(y, sr)
    assert m.shape == (3, C.N_MELS, C.TARGET_T)
    assert m.dtype == np.float32 and m.min() >= 0 and m.max() <= 1


@pytest.mark.parametrize("t", [500, C.TARGET_T, 900])
def test_crop_or_pad(t):
    x = np.ones((3, 4, t), np.float32)
    out = crop_or_pad(x)
    assert out.shape[-1] == C.TARGET_T
    if t < C.TARGET_T:  # symmetric zero padding
        assert out[..., 0].sum() == 0 and out[..., -1].sum() == 0


def test_checkpoint_layout_matches_original(model):
    keys = list(model.state_dict())
    assert len(keys) == 34
    assert keys[0] == "encoder.features.0.weight" and keys[-1] == "head.2.bias"


def test_forward_and_gradcam(model):
    a, b = torch.rand(3, C.N_MELS, C.TARGET_T), torch.rand(3, C.N_MELS, C.TARGET_T)
    assert model(a[None], b[None]).shape == (1,)
    ha, hb, p = SiameseGradCAM(model)(a, b)
    assert ha.shape == hb.shape == (C.N_MELS, C.TARGET_T)
    assert 0 <= p <= 1 and 0 <= ha.min() and ha.max() <= 1


@pytest.fixture
def tiny_split(tmp_path):
    rows = []
    for t in range(3):
        for v in C.VERSIONS:
            for c in range(4):
                if t == 0 and c == 0 and v != "original":
                    continue  # segment (T0, 0) exists in one version only
                f = tmp_path / f"T{t}_{v}_{c}.npy"
                np.save(f, np.random.rand(3, C.N_MELS, 600).astype(np.float16))
                rows.append(dict(track_id=f"T{t}", version=v, chunk_id=str(c), path=str(f)))
    return pd.DataFrame(rows)


def test_fixed_pairs_are_valid_and_deterministic(tiny_split):
    p1, p2 = FixedPairs(tiny_split, seed=1), FixedPairs(tiny_split, seed=1)
    assert p1.pairs == p2.pairs
    df = tiny_split
    for i, j, y in p1.pairs:
        assert i != j
        if y == 1:
            assert df.track_id[i] == df.track_id[j] and df.chunk_id[i] == df.chunk_id[j]
            assert df.version[i] != df.version[j]
        else:
            assert df.track_id[i] != df.track_id[j]
    a, b, y = p1[0]
    assert a.shape == b.shape == (3, C.N_MELS, C.TARGET_T)


def test_train_pairs_skip_single_version_anchors(tiny_split):
    ds = TrainPairs(tiny_split)
    single = tiny_split.index[(tiny_split.track_id == "T0") & (tiny_split.chunk_id == "0")][0]
    assert single not in ds.anchors


def test_mcic_helpers(model):
    assert normalize_filename("010_b_While_We_Dream.mid ") == "010_b_While_We_Dream.mid"
    assert normalize_filename("107_a_Hachigatsu_no_Rainy_[August_Rainy].mid") == "107_a_Hachigatsu_no_Rainy_August_Rainy.mid"
    song = np.random.rand(3, C.N_MELS, 2000).astype(np.float32)
    assert len(windows(song)) == 5
    for mode in ("center", "max-window"):
        assert 0 <= score_pair(model, song, song[..., ::-1].copy(), mode) <= 1


def test_melodysim_metadata():
    assert parse_relative_path("test/Track01962/version_2/1.wav") == ("test", "Track01962", "version_2", "1")
    assert len(read_shard_list()) == 103
    assert len(read_shard_list(split="test")) == 35
