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
