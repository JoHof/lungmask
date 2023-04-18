import os

import numpy as np
import pydicom as pyd
import pytest

from lungmask.mask import LMInferer, apply, apply_fused
from lungmask.utils import read_dicoms


@pytest.fixture(scope="session")
def fixture_testvol():
    return read_dicoms(os.path.join(os.path.dirname(__file__), "testdata"))[0]


def test_apply(fixture_testvol):
    res = apply(fixture_testvol)
    assert np.all(np.unique(res, return_counts=True)[1] == [423000, 64752, 36536])


def test_apply_fused(fixture_testvol):
    res = apply_fused(fixture_testvol)
    assert np.all(
        np.unique(res, return_counts=True)[1] == [423000, 13334, 23202, 23834, 40918]
    )
