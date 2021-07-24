import scipy.ndimage as ndimage
import skimage.measure
import numpy as np
from torch.utils.data import Dataset
import os
import sys
import SimpleITK as sitk
import pydicom as pyd
import logging
from tqdm import tqdm
import fill_voids
import skimage.morphology

class ImageLoader:
    def __init__(self, in_path=None):
        """Class for loading image data"""
        self.in_path = in_path

    def get_input_image(self):
        if os.path.isfile(self.in_path):
            logging.info(f'Read input: {path}')
            input_image = sitk.ReadImage(self.in_path)
        else:
            logging.info(f'Looking for dicoms in {path}')
            dicom_vols = self._read_dicoms(original=False, primary=False)
            if len(dicom_vols) < 1:
                sys.exit('No dicoms found!')
            if len(dicom_vols) > 1:
                logging.warning("There are more than one volume in the path, will take the largest one")
            input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
        return input_image

    def _read_dicoms(self, primary=True, original=True):
        """function only used by get_input_image. set as private"""
        allfnames = []
        for dir, _, fnames in os.walk(self.in_path):
            [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

        dcm_header_info = []
        dcm_parameters = []
        unique_set = []  # need this because too often there are duplicates of dicom files with different names
        i = 0
        for fname in tqdm(allfnames):
            filename_ = os.path.splitext(os.path.split(fname)[1])
            i += 1
            if filename_[0] != 'DICOMDIR':
                try:
                    dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                    if dicom_header is not None:
                        if 'ImageType' in dicom_header:
                            if primary:
                                is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                            else:
                                is_primary = True

                            if original:
                                is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                            else:
                                is_original = True

                            # if 'ConvolutionKernel' in dicom_header:
                            #     ck = dicom_header.ConvolutionKernel
                            # else:
                            #     ck = 'unknown'
                            if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                                h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                                  dicom_header.ImagePositionPatient]
                                h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                          dicom_header.ImagePositionPatient]
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
        dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
        # dcm_parameters = np.asarray(dcm_parameters)[sidx]
        vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
        n_vol = len(vol_unique[1])
        logging.info('There are ' + str(n_vol) + ' volumes in the study')

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


class ImageSaver:
    def __init__(self, out_path=None):
        """Class for saving data"""
        self.out_path = out_path

    def save_results(self, result, img):
        result_out = sitk.GetImageFromArray(result)
        result_out.CopyInformation(img)
        logging.info(f'Save result to: {self.out_path }')
        sitk.WriteImage(result_out,self.out_path)

class ImagePreProcessor:
    def __init__(self):
        """Class for preprocessing data"""
        pass

    def preprocess(self, img, label=None, resolution=[192, 192]):
        imgmtx = np.copy(img)
        lblsmtx = np.copy(label)

        imgmtx[imgmtx < -1024] = -1024
        imgmtx[imgmtx > 600] = 600
        cip_xnew = []
        cip_box = []
        cip_mask = []
        for i in range(imgmtx.shape[0]):
            if label is None:
                (im, m, box) = self._crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
            else:
                (im, m, box) = self._crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0],
                                               height=resolution[1])
                cip_mask.append(m)
            cip_xnew.append(im)
            cip_box.append(box)
        if label is None:
            return np.asarray(cip_xnew), cip_box
        else:
            return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)

    def _crop_and_resize(self, img, mask=None, width=192, height=192):
        """utility function only used by preprocessing. Set as private"""
        bmask = self._simple_bodymask(img)
        # img[bmask==0] = -1024 # this line removes background outside of the lung.
        # However, it has been shown problematic with narrow circular field of views that touch the lung.
        # Possibly doing more harm than help
        reg = skimage.measure.regionprops(skimage.measure.label(bmask))
        if len(reg) > 0:
            bbox = np.asarray(reg[0].bbox)
        else:
            bbox = (0, 0, bmask.shape[0], bmask.shape[1])
        img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
        if not mask is None:
            mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
            # mask = ndimage.binary_closing(mask,iterations=5)
        return img, mask, bbox

    def _simple_bodymask(self, img):
        """utility function only used by crop_and_resize. Set as private"""
        maskthreshold = -500
        oshape = img.shape
        img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
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
        real_scaling = np.asarray(oshape)/128
        return ndimage.zoom(bodymask, real_scaling, order=0)



class ImagePostProcessor:
    def __init__(self):
        """Class for postprocessing data"""
        pass

    def postprocessing(self, label_image, spare=[]):
        '''some post-processing mapping small label patches to the neighbout whith which they share the
            largest border. All connected components smaller than min_area will be removed
        '''

        # merge small components to neighbours
        regionmask = skimage.measure.label(label_image)
        origlabels = np.unique(label_image)
        origlabels_maxsub = np.zeros((max(origlabels) + 1,),
                                     dtype=np.uint32)  # will hold the largest component for a label
        regions = skimage.measure.regionprops(regionmask, label_image)
        regions.sort(key=lambda x: x.area)
        regionlabels = [x.label for x in regions]

        # will hold mapping from regionlabels to original labels
        region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
        for r in regions:
            if r.area > origlabels_maxsub[r.max_intensity]:
                origlabels_maxsub[r.max_intensity] = r.area
                region_to_lobemap[r.label] = r.max_intensity

        for r in tqdm(regions):
            if (r.area < origlabels_maxsub[
                r.max_intensity] or r.max_intensity in spare) and r.area > 2:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
                bb = self._bbox_3D(regionmask == r.label)
                sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
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
                if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                    regions[regionlabels.index(mapto)].max_intensity]:
                    origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
                regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

        outmask_mapped = region_to_lobemap[regionmask]
        outmask_mapped[outmask_mapped == spare] = 0

        if outmask_mapped.shape[0] == 1:
            # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
            holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
        else:
            holefiller = fill_voids.fill

        outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
        for i in np.unique(outmask_mapped)[1:]:
            outmask[holefiller(self._keep_largest_connected_component(outmask_mapped == i))] = i

        return outmask

    def _bbox_3D(self, labelmap, margin=2):
        """utility function only used by postprocessing. Set as private"""
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

    def _keep_largest_connected_component(self,mask):
        """utility function only used by postprocessing. Set as private"""
        mask = skimage.measure.label(mask)
        regions = skimage.measure.regionprops(mask)
        resizes = np.asarray([x.area for x in regions])
        max_region = np.argsort(resizes)[-1] + 1
        mask = mask == max_region
        return mask


class ImageDataHandler:
    def __init__(self, in_path=None, out_path=None):
        """
        an aggregate of images and image pre/postprocessing behavior.

        :param in_path: input file path to load image data
        :param out_path: output file path to save image data
        """
        self.in_path = in_path
        self.out_path = out_path
        self.image_loader = ImageLoader(in_path)
        self.image_preprocessor = ImagePreProcessor()
        self.image_postprocessor = ImagePostProcessor()
        self.image_saver = ImageSaver(out_path)

    def get_input_image(self):
        return self.image_loader.get_input_image()

    def save_results(self, result, img):
        self.image_saver.save_results(result, img)

    def preprocess(self, img, label=None, resolution=[192, 192]):
        return self.image_preprocessor.preprocess(img, label, resolution)

    def postprocessing(self, label_image, spare=[]):
        return self.image_postprocessor.postprocessing(label_image, spare)