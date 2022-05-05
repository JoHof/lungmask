import logging
import os
import sys
from typing import Tuple, Union

import cv2
import fill_voids
import numpy as np
import pydicom as pyd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure
import skimage.morphology
from torch.utils.data import Dataset
from tqdm import tqdm

ORDER2OCVINTER = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA, 3: cv2.INTER_CUBIC}


def preprocess(img, label=None, resolution=[192, 192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx < -1024] = -1024
    imgmtx[imgmtx > 600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(
                imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0], height=resolution[1]
            )
            cip_mask.append(m)
        cip_xnew.append(im)
        cip_box.append(box)
    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)


def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    # PATCH: order is linear to reproduce behaviour before interpolation bug
    # Set order to 0 after current version is no longer needed
    img = cv2_zoom(img, 128 / np.asarray(img.shape), order=1)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape) / 128
    # PATCH: order is linear to reproduce behaviour before interpolation bug
    # Set order to 0 after current version is no longer needed
    return cv2_zoom(bodymask, real_scaling, order=1)


def crop_and_resize(img, mask=None, width=192, height=192):
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    img = cv2_zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)

    if not mask is None:
        mask = mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask = cv2_zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


## For some reasons skimage.transform leads to edgy mask borders compared to ndimage.zoom
# def reshape_mask(mask, tbox, origsize):
#     res = np.ones(origsize) * 0
#     resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
#     imgres = skimage.transform.resize(mask, resize, order=0, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)
#     res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
#     return res


def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    # Change order 0 (nearest) to 2 (area)
    imgres = cv2_zoom(mask, resize / np.asarray(mask.shape), order=0, pseudo_linear=True)
    res[tbox[0] : tbox[2], tbox[1] : tbox[3]] = imgres
    return res


class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :].astype(np.float)


def read_dicoms(path, primary=True, original=True):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    dcm_parameters = []
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != "DICOMDIR":
            try:
                dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                if dicom_header is not None:
                    if "ImageType" in dicom_header:
                        if primary:
                            is_primary = all([x in dicom_header.ImageType for x in ["PRIMARY"]])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ["ORIGINAL"]])
                        else:
                            is_original = True

                        # if 'ConvolutionKernel' in dicom_header:
                        #     ck = dicom_header.ConvolutionKernel
                        # else:
                        #     ck = 'unknown'
                        if is_primary and is_original and "LOCALIZER" not in dicom_header.ImageType:
                            h_info_wo_name = [
                                dicom_header.StudyInstanceUID,
                                dicom_header.SeriesInstanceUID,
                                dicom_header.ImagePositionPatient,
                            ]
                            h_info = [
                                dicom_header.StudyInstanceUID,
                                dicom_header.SeriesInstanceUID,
                                fname,
                                dicom_header.ImagePositionPatient,
                            ]
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)
                                # kvp = None
                                # if 'KVP' in dicom_header:
                                #     kvp = dicom_header.KVP
                                # dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)

    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    logging.info("There are " + str(n_vol) + " volumes in the study")

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]
        relevant_series.append(vol_files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        relevant_volumes.append(vol)

    return relevant_volumes


def get_input_image(path):
    if os.path.isfile(path):
        logging.info(f"Read input: {path}")
        input_image = sitk.ReadImage(path)
    else:
        logging.info(f"Looking for dicoms in {path}")
        dicom_vols = read_dicoms(path, original=False, primary=False)
        if len(dicom_vols) < 1:
            sys.exit("No dicoms found!")
        if len(dicom_vols) > 1:
            logging.warning("There are more than one volume in the path, will take the largest one")
        input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
    return input_image


def postrocessing(label_image, spare=[], verbose=True):
    """some post-processing mapping small label patches to the neighbout whith which they share the
    largest border. All connected components smaller than min_area will be removed
    """

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        if r.area > origlabels_maxsub[r.max_intensity]:
            origlabels_maxsub[r.max_intensity] = r.area
            region_to_lobemap[r.label] = r.max_intensity

    for r in tqdm(regions, disable=not verbose):
        if (
            r.area < origlabels_maxsub[r.max_intensity] or r.max_intensity in spare
        ) and r.area > 2:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n != spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if (
                regions[regionlabels.index(mapto)].area
                == origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity]
            ):
                origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
            regions[regionlabels.index(mapto)].__dict__["_cache"]["area"] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped == spare] = 0

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax

    if rmax - rmin == 0:
        rmax = rmin + 1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


def cv2_zoom(
    img: np.ndarray, scale: Union[Tuple[float, float], float], order: int = 0, pseudo_linear: bool = False, lin_thr: float = 0.3
) -> np.ndarray:
    """
    Arguments
    ---------
        img             (h, w[, ch]) : [num] 0-channel or multichannel image.
        scale   (scale_x[, scale_y]) : Scale for interpolation.
        order                  (int) : Numeral encoding of interpolation order.
        pseudo_linear         (bool) : If True, apply linear interpolation and value thresholding 
                                       to each individual mask value.
        lin_thr              (float) : (Only used if `pseudo_linear=True`) After linear interpolation 
                                       all values above `lin_thr` are set to the mask value.

    Returns
    -------
        out_img     (h_o, w_o[, ch]) : Interpolated image.

    Interpolates 2D images.
    """
    assert order in ORDER2OCVINTER, f"Only order from dict {ORDER2OCVINTER} are supported"

    if isinstance(scale, float):
        scale = (scale, scale)

    if img.dtype == np.bool:
        img = img.astype(np.uint8)
    if np.issubdtype(img.dtype, np.integer) and img.dtype != np.uint8:
        img = img.astype(np.float64)

    out_shape = tuple((np.asarray(img.shape[:2]) * np.asarray(scale)).round().astype(int)[::-1])

    if pseudo_linear:
        uniques = np.unique(img)
        out_shape_with_channels = list(out_shape[::-1]) + list(img.shape[2:])
        out_img = np.zeros(out_shape_with_channels, dtype=img.dtype)
        for value in uniques[uniques != 0]:
            img_c = (img == value).astype(img.dtype) * value
            output = cv2.resize(img_c, out_shape, interpolation=cv2.INTER_LINEAR)
            output[output >= value * lin_thr] = value
            output[output != value] = 0
            out_img[output == value] = output[output == value]
    else:
        out_img = cv2.resize(img, out_shape, interpolation=ORDER2OCVINTER[order])

    return out_img
