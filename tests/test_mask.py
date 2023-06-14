import os

import numpy as np
import pydicom as pyd
import pytest

from lungmask.mask import LMInferer, apply, apply_fused
from lungmask.utils import read_dicoms


@pytest.fixture(scope="session")
def fixture_testvol():
    return read_dicoms(os.path.join(os.path.dirname(__file__), "testdata"))[0]


def test_LMInferer(fixture_testvol):
    inferer = LMInferer(
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
