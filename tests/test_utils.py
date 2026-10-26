import os

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from lungmask.utils import (
    bbox_3D,
    crop_and_resize,
    keep_largest_connected_component,
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


@pytest.mark.parametrize("read_metadata,exp_len_metadata", [(True, 22), (False, 0)])
def test_read_dicoms(read_metadata, exp_len_metadata):
    d = read_dicoms(
        os.path.join(os.path.dirname(__file__), "testdata"), read_metadata=read_metadata
    )
    assert d[0].GetSize() == (512, 512, 2)
    assert len(d[0].GetMetaDataKeys()) == exp_len_metadata


@pytest.mark.parametrize(
    "orientation", [(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 1), (1, 0, 0, 0, 0, -1)]
)
def test_read_dicoms_sorts_along_slice_normal(tmp_path, monkeypatch, orientation):
    normal = np.cross(orientation[:3], orientation[3:])
    source = os.path.join(os.path.dirname(__file__), "testdata", "0.dcm")
    # Deliberately supply slices out of spatial order, including non-axial series.
    for file_index, position in enumerate([2, 0, 1]):
        dicom = pydicom.dcmread(source)
        dicom.ImageOrientationPatient = list(orientation)
        dicom.ImagePositionPatient = (normal * position).tolist()
        dicom.SOPInstanceUID = pydicom.uid.generate_uid()
        dicom.file_meta.MediaStorageSOPInstanceUID = dicom.SOPInstanceUID
        dicom.RescaleSlope = 1
        dicom.RescaleIntercept = 0
        dicom.PixelData = np.full(
            (dicom.Rows, dicom.Columns), position, dtype=np.int16
        ).tobytes()
        dicom.save_as(tmp_path / f"{file_index}.dcm")

    monkeypatch.setattr(
        "lungmask.utils.os.walk",
        lambda path: [(str(tmp_path), [], ["0.dcm", "1.dcm", "2.dcm"])],
    )
    volume = read_dicoms(tmp_path, disable_tqdm=True)[0]

    assert np.array_equal(sitk.GetArrayFromImage(volume)[:, 0, 0], [0, 1, 2])
    assert np.allclose(volume.GetOrigin(), [0, 0, 0])
    assert np.allclose(np.asarray(volume.GetDirection()).reshape(3, 3)[:, 2], normal)


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


@pytest.mark.parametrize("read_metadata,exp_len_metadata", [(True, 22), (False, 0)])
def test_load_input_image(tmp_path, read_metadata, exp_len_metadata):
    # test dicom
    d = load_input_image(
        os.path.join(os.path.dirname(__file__), "testdata"), read_metadata=read_metadata
    )
    assert d.GetSize() == (512, 512, 2)
    assert len(d.GetMetaDataKeys()) == exp_len_metadata

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


@pytest.mark.parametrize("extra_spare", [False, True])
def test_postprocessing_does_not_merge_lobe_fragment_into_spare(extra_spare):
    label_image = np.zeros((2, 6, 17), dtype=np.uint8)
    label_image[:, 1, 1:4] = 1  # largest component of lobe 1
    label_image[:, 1:3, 8:11] = 6  # spare region, with component ID 2
    label_image[:, 1:3, 11:15] = 3
    label_image[:, 3, 8:10] = 1  # fragment shares a larger border with spare
    label_image[:, 4, 8:9] = 2  # valid neighbor for the fragment
    if extra_spare:
        # A tied spare component can have a zero entry in region_to_lobemap.
        label_image[:, 1:3, 4:7] = 6

    result = postprocessing(label_image, spare=[6], disable_tqdm=True)

    assert np.all(result[:, 3, 8:10] == 2)
    assert np.all(result[:, 1:3, 8:11] == 3)
    if extra_spare:
        assert np.all(result[:, 1:3, 4:7] == 1)


@pytest.mark.parametrize("slices", [1, 2])
@pytest.mark.parametrize("label_count", [1, 2])
def test_postprocessing_preserves_labels_without_background(slices, label_count):
    label_image = np.ones((slices, 4, 4), dtype=np.uint8)
    if label_count == 2:
        label_image[:, :, 2:] = 2

    result = postprocessing(label_image, disable_tqdm=True)

    assert np.array_equal(result, label_image)


def test_keep_largest_connected_component_empty():
    mask = np.zeros((2, 4, 4), dtype=bool)

    result = keep_largest_connected_component(mask)

    assert result.dtype == bool
    assert np.array_equal(result, mask)


def test_postprocessing_single_slice_fills_only_enclosed_holes():
    label_image = np.zeros((1, 6, 6), dtype=np.uint8)
    label_image[:, 1:5, 1:5] = 1
    label_image[:, 2:4, 2:4] = 0
    expected = label_image.copy()
    expected[:, 2:4, 2:4] = 1

    result = postprocessing(label_image, disable_tqdm=True)

    assert np.array_equal(result, expected)
