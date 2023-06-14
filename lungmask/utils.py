import logging
import os
import sys
from typing import Tuple

import fill_voids
import numpy as np
import pydicom as pyd
import SimpleITK as sitk
import skimage.measure
import skimage.morphology
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess(
    img: np.ndarray, resolution: list = [192, 192]
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses the image by clipping, cropping and resizing. Clipping at -1024 and 600 HU, cropping to the body

    Args:
        img (np.ndarray): Image to be preprocessed
        resolution (list, optional): Target size after preprocessing. Defaults to [192, 192].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed image and the cropping bounding box
    """
    imgmtx = np.copy(img)
    imgmtx = np.clip(imgmtx, -1024, 600)
    cip_xnew = []
    cip_box = []
    for imslice in imgmtx:
        im, box = crop_and_resize(imslice, width=resolution[0], height=resolution[1])
        cip_xnew.append(im)
        cip_box.append(box)
    return np.asarray(cip_xnew), cip_box


def simple_bodymask(img: np.ndarray) -> np.ndarray:
    """Computes a simple bodymask by thresholding the image at -500 HU and then filling holes and removing small objects

    Args:
        img (np.ndarray): CT image (single slice) in HU

    Returns:
        np.ndarray: Binary mask of the body
    """

    # Here are some heuristics to get a body mask
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128 / np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(
        int
    )
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape) / 128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(
    img: np.ndarray, width: int = 192, height: int = 192
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops the image to the body and resizes it to the specified size

    Args:
        img (np.ndarray): Image to be cropped and resized
        width (int, optional): Target width to be resized to. Defaults to 192.
        height (int, optional): Target height to be resized to. Defaults to 192.

    Returns:
        Tuple[np.ndarray, np.ndarray]: resized image and the cropping bounding box
    """
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
    img = ndimage.zoom(
        img, np.asarray([width, height]) / np.asarray(img.shape), order=1
    )
    return img, bbox


def reshape_mask(mask: np.ndarray, tbox: np.ndarray, origsize: tuple) -> np.ndarray:
    """Reshapes the mask to the original size given bounding box and original size

    Args:
        mask (np.ndarray): Mask to be resampled (nearest neighbor)
        tbox (np.ndarray): Bounding box in original image covering field of view of the mask
        origsize (tuple): Original images size

    Returns:
        np.ndarray: Resampled mask in original image space
    """
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0] : tbox[2], tbox[1] : tbox[3]] = imgres
    return res


def read_dicoms(path, primary=True, original=True, disable_tqdm=False):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    unique_set = (
        []
    )  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames, disable=disable_tqdm):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != "DICOMDIR":
            try:
                dicom_header = pyd.dcmread(
                    fname, defer_size=100, stop_before_pixels=True, force=True
                )
                if dicom_header is not None:
                    if "ImageType" in dicom_header:
                        if primary:
                            is_primary = all(
                                [x in dicom_header.ImageType for x in ["PRIMARY"]]
                            )
                        else:
                            is_primary = True

                        if original:
                            is_original = all(
                                [x in dicom_header.ImageType for x in ["ORIGINAL"]]
                            )
                        else:
                            is_original = True

                        if (
                            is_primary
                            and is_original
                            and "LOCALIZER" not in dicom_header.ImageType
                        ):
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

            except Exception as e:
                logging.error("Unexpected error:", e)
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)

    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    if n_vol == 1:
        logging.info("There is " + str(n_vol) + " volume in the study")
    else:
        logging.info("There are " + str(n_vol) + " volumes in the study")

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray(
            [np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]]
        )
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]
        relevant_series.append(vol_files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        relevant_volumes.append(vol)

    return relevant_volumes


def load_input_image(path: str, disable_tqdm=False) -> sitk.Image:
    """Loads image, if path points to a file, file will be loaded. If path points ot a folder, a DICOM series will be loaded. If multiple series are present, the largest series (higher number of slices) will be loaded.

    Args:
        path (str): File or folderpath to be loaded. If folder, DICOM series is expected
        disable_tqdm (bool, optional): Disable tqdm progress bar. Defaults to False.

    Returns:
        sitk.Image: Loaded image
    """
    if os.path.isfile(path):
        logging.info(f"Read input: {path}")
        input_image = sitk.ReadImage(path)
    else:
        logging.info(f"Looking for dicoms in {path}")
        dicom_vols = read_dicoms(
            path, original=False, primary=False, disable_tqdm=disable_tqdm
        )
        if len(dicom_vols) < 1:
            sys.exit("No dicoms found!")
        if len(dicom_vols) > 1:
            logging.warning(
                "There are more than one volume in the path, will take the largest one"
            )
        input_image = dicom_vols[
            np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)
        ]
    return input_image


def postprocessing(
    label_image: np.ndarray,
    spare: list = [],
    disable_tqdm: bool = False,
    skip_below: int = 3,
) -> np.ndarray:
    """some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. Only largest connected components (CC) for each label will be kept. If a label is member of the spare list it will be mapped to neighboring labels and not present in the final labelling.

    Args:
        label_image (np.ndarray): Label image (int) to be processed
        spare (list, optional): Labels that are used for mapping to neighbors but not considered for final labelling. This is used for label fusion with a filling model. Defaults to [].
        disable_tqdm (bool, optional): If true, tqdm will be diabled. Defaults to False.
        skip_below (int, optional): If a CC is smaller than this value. It will not be merged but removed. This is for performance optimization.

    Returns:
        np.ndarray: Postprocessed volume
    """
    logging.info("Postprocessing")

    # CC analysis
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros(
        (max(origlabels) + 1,), dtype=np.uint32
    )  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions, disable=disable_tqdm):
        r_max_intensity = int(r.max_intensity)
        if (
            r.area < origlabels_maxsub[r_max_intensity] or r_max_intensity in spare
        ) and r.area >= skip_below:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n not in spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto

            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if (
                regions[regionlabels.index(mapto)].area
                == origlabels_maxsub[
                    int(regions[regionlabels.index(mapto)].max_intensity)
                ]
            ):
                origlabels_maxsub[
                    int(regions[regionlabels.index(mapto)].max_intensity)
                ] += myarea
            regions[regionlabels.index(mapto)].__dict__["_cache"]["area"] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[np.isin(outmask_mapped, spare)] = 0

    if outmask_mapped.shape[0] == 1:
        holefiller = (
            lambda x: skimage.morphology.area_closing(
                x[0].astype(int), area_threshold=64
            )[None, :, :]
            == 1
        )
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    """Compute bounding box of a 3D labelmap.

    Args:
        labelmap (np.ndarray): Input labelmap
        margin (int, optional): Margin to add to the bounding box. Defaults to 2.

    Returns:
        np.ndarray: Bounding box as [zmin, zmax, ymin, ymax, xmin, xmax]
    """
    shape = labelmap.shape
    dimensions = np.arange(len(shape))
    bmins = []
    bmaxs = []
    margin = [margin] * len(dimensions)
    for dim, dim_margin, dim_shape in zip(dimensions, margin, shape):
        margin_label = np.any(labelmap, axis=tuple(dimensions[dimensions != dim]))
        bmin, bmax = np.where(margin_label)[0][[0, -1]]
        bmin -= dim_margin
        bmax += dim_margin + 1
        bmin = max(bmin, 0)
        bmax = min(bmax, dim_shape)
        bmins.append(bmin)
        bmaxs.append(bmax)

    bbox = np.array(list(zip(bmins, bmaxs))).flatten()
    return bbox


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keeps largest connected component (CC)

    Args:
        mask (np.ndarray): Input label map

    Returns:
        np.ndarray: Binary label map with largest CC
    """
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask
