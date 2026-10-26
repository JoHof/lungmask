import os
import shutil

import numpy as np
import pytest
import torch

from lungmask.mask import MODEL_URLS, LMInferer
from lungmask.utils import read_dicoms


@pytest.fixture(scope="session")
def fixture_testvol():
    return read_dicoms(os.path.join(os.path.dirname(__file__), "testdata"))[0]


@pytest.fixture(scope="session")
def fixture_weights_path_R231(tmpdir_factory):
    # we make sure the model is there
    torch.hub.load_state_dict_from_url(
        MODEL_URLS["R231"][0], progress=True, map_location=torch.device("cpu")
    )
    modelbasename = os.path.basename(MODEL_URLS["R231"][0])
    modelpath = os.path.join(torch.hub.get_dir(), "checkpoints", modelbasename)
    tmppath = str(tmpdir_factory.mktemp("weights").join(modelbasename))
    shutil.copy(modelpath, tmppath)
    return tmppath


def test_LMInferer(fixture_testvol, fixture_weights_path_R231):
    inferer = LMInferer(
        force_cpu=True,
        tqdm_disable=True,
    )
    res = inferer.apply(fixture_testvol)
    assert np.all(np.unique(res, return_counts=True)[1] == [423000, 64752, 36536])

    # here, we provide a path to the R231 weights but specify LTRCLobes (6 channel) as modelname
    # The modelname should be ignored and a 3 channel output should be generated
    inferer = LMInferer(
        modelname="LTRCLobes",
        modelpath=fixture_weights_path_R231,
        force_cpu=True,
        tqdm_disable=True,
    )
    res = inferer.apply(fixture_testvol)
    assert np.all(np.unique(res, return_counts=True)[1] == [423000, 64752, 36536])


def test_LMInferer_fused(fixture_testvol):
    inferer = LMInferer(
        modelname="LTRCLobes",
        force_cpu=True,
        fillmodel="R231",
        tqdm_disable=True,
    )
    res = inferer.apply(fixture_testvol)
    assert np.all(
        np.unique(res, return_counts=True)[1] == [423000, 13334, 23202, 23834, 40918]
    )


def test_LMInferer_fused_preserves_fill_only_voxels(monkeypatch):
    monkeypatch.setattr("lungmask.mask.get_model", lambda *args: torch.nn.Identity())
    inferer = LMInferer(
        modelname="LTRCLobes", fillmodel="R231", force_cpu=True, tqdm_disable=True
    )
    lobes = np.zeros((2, 5, 32), dtype=np.uint8)
    for value, start in [(1, 1), (2, 7), (3, 13), (4, 19)]:
        lobes[:, 1, start : start + 3] = value
    lobes[:, 2, 26:30] = 5
    fill = (lobes > 0).astype(np.uint8)
    fill[:, 1, 26:28] = 1
    # Spare label 6 is component 5; its valid lobe-5 neighbor is component 6.
    monkeypatch.setattr(
        inferer,
        "_inference",
        lambda image, model: lobes.copy() if model is inferer.model else fill.copy(),
    )

    result = inferer.apply(np.zeros(lobes.shape, dtype=np.int16))

    expected = lobes.copy()
    expected[:, 1, 26:28] = 5
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("model_kind", ["base", "fill"])
def test_LMInferer_custom_path_overrides_unknown_model_name(monkeypatch, model_kind):
    calls = []

    def get_stub_model(name, path):
        calls.append((name, path))
        return torch.nn.Identity()

    monkeypatch.setattr("lungmask.mask.get_model", get_stub_model)
    if model_kind == "base":
        LMInferer(modelname="custom", modelpath="/models/base.pth", force_cpu=True)
        assert calls == [("base.pth", "/models/base.pth")]
    else:
        LMInferer(
            fillmodel="custom", fillmodel_path="/models/fill.pth", force_cpu=True
        )
        assert calls == [("R231", None), ("fill.pth", "/models/fill.pth")]
