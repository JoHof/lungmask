import os
import sys
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from lungmask.__main__ import main


def test_main(tmp_path):
    testdata_path = os.path.join(os.path.dirname(__file__), "testdata")
    testdata_path_out = str(tmp_path / "testres.nii.gz")
    argv = ["__main__.py", testdata_path, testdata_path_out]
    with patch.object(sys, "argv", argv):
        main()

    assert os.path.exists(testdata_path_out)
    res = sitk.GetArrayFromImage(sitk.ReadImage(testdata_path_out))
    assert np.all(np.unique(res, return_counts=True)[1] == [423000, 64752, 36536])
