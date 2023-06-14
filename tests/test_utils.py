import os

import numpy as np
import SimpleITK as sitk

from lungmask.utils import (
    bbox_3D,
    crop_and_resize,
    load_input_image,
    postprocessing,
    preprocess,
    read_dicoms,
    reshape_mask,
    simple_bodymask,
)

# creating test dicom data for reference
# import pydicom as pd
# import pydicom as pyd
# from pydicom.dataset import FileMetaDataset
#
# studyuid = pyd.uid.generate_uid()
# seriesuid = pyd.uid.generate_uid()
# for i in range(2):
#     testd = pyd.Dataset()
#     testd.PixelSpacing = [.625, .625]
#     testd.SliceThickness = 1
#     testd.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
#     testd.ImagePositionPatient = [0, 0, float(i)]
#     testd.RescaleIntercept = 0
#     testd.RescaleSlope = 1
#     testd.SeriesInstanceUID = studyuid
#     testd.StudyInstanceUID = seriesuid
#     testd.PixelData = v[60]
#     testd.Rows = 512
#     testd.Columns = 512
#     testd.SamplesPerPixel = 1
#     testd.BitsAllocated = 16
#     testd.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
#     testd.SOPInstanceUID = pyd.uid.generate_uid()
#     testd.PixelRepresentation = 1
#     testd.BitsStored = 16
#     testd.SpecificCharacterSet = 'ISO_IR 100'
#     testd.PhotometricInterpretation = 'MONOCHROME2'
#     testd.SOPClassUID = pyd.uid.CTImageStorage
#     testd.StorageMediaFileSetUID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.211790042620307056609660772296"

#     fmd = FileMetaDataset()
#     fmd.FileMetaInformationVersion = b'\x00\x01'
#     fmd.MediaStorageSOPClassUID = pyd.uid.CTImageStorage
#     fmd.MediaStorageSOPInstanceUID = pyd.uid.generate_uid()
#     fmd.TransferSyntaxUID = pyd.uid.ExplicitVRLittleEndian
#     fmd.ImplementationClassUID = '1.2.40.0.13.1.1.1'
#     fmd.ImplementationVersionName = 'PYDICOM 1.4.2'


def test_bbox_3D():
    m = np.zeros((10, 10, 10), dtype=np.uint8)
    m[2:8, 3:7, 4:6] = 1

    bb = bbox_3D(m, margin=2)
    assert tuple(bb) == (0, 10, 1, 9, 2, 8)


def test_read_dicoms():
    d = read_dicoms(os.path.join(os.path.dirname(__file__), "testdata"))
    assert d[0].GetSize() == (512, 512, 2)


def test_simple_bodymask():
    img = np.full((10, 10), dtype=np.int16, fill_value=-1000)
    img[2:8, 3:7] = 1
    img[9, 9] = 1
    mask = simple_bodymask(img)
    assert np.sum(mask) == 24


def test_crop_and_resize():
    img = np.full((10, 10), dtype=np.int16, fill_value=-1000)
    img[2:8, 3:7] = 1
    img[9, 9] = 1
    cropped, bb = crop_and_resize(img, width=20, height=20)
    assert tuple(bb) == (2, 3, 8, 7)
    assert cropped.shape == (20, 20)
    assert np.sum(cropped) == 400


def test_preprocess():
    img = np.full((2, 10, 10), dtype=np.int16, fill_value=-1000)
    img[:, 2:8, 3:7] = 1
    img[:, 9, 9] = 1
    cropped, bb = preprocess(img, resolution=[20, 20])
    for sl, bb_ in zip(cropped, bb):
        assert tuple(bb_) == (2, 3, 8, 7)
        assert sl.shape == (20, 20)
        assert np.sum(sl) == 400


def test_reshape_mask():
    msk = np.full((10, 10), dtype=np.uint8, fill_value=1)
    bb = (2, 2, 22, 22)
    cropped_mask = reshape_mask(msk, bb, origsize=(30, 30))
    assert cropped_mask.shape == (30, 30)
    assert np.sum(cropped_mask) == 400


def test_load_input_image(tmp_path):
    # test dicom
    d = load_input_image(os.path.join(os.path.dirname(__file__), "testdata"))
    assert d.GetSize() == (512, 512, 2)

    # test nifti
    fp_testnii = str(tmp_path / "test.nii.gz")
    sitk.WriteImage(d, fp_testnii)
    d = load_input_image(fp_testnii)
    assert d.GetSize() == (512, 512, 2)


def test_postprocessing():
    label_image = np.zeros((1, 6, 6), dtype=np.uint8)
    label_image[0] = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 2, 0, 3, 1, 0],
            [0, 4, 4, 4, 0, 0],
            [0, 4, 0, 4, 0, 0],
            [0, 4, 4, 4, 0, 0],
        ]
    )

    res_gt = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 2, 2, 0],
        [0, 1, 0, 3, 2, 0],
        [0, 4, 4, 4, 0, 0],
        [0, 4, 0, 4, 0, 0],
        [0, 4, 4, 4, 0, 0],
    ]

    res = postprocessing(
        np.tile(label_image, (2, 1, 1)), spare=[], disable_tqdm=True, skip_below=1
    )[0]
    assert np.all(res == res_gt)

    res = postprocessing(
        np.tile(label_image, (2, 1, 1)), spare=[3], disable_tqdm=True, skip_below=1
    )[0]
    assert res[2, 3] == 2

    res = postprocessing(
        np.tile(label_image, (2, 1, 1)), spare=[3], disable_tqdm=True, skip_below=3
    )[0]
    assert res[2, 1] == 0
