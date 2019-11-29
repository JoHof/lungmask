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


def preprocess(img, label=None, resolution = [192,192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx<-1024] = -1024
    imgmtx[imgmtx>600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im,m,box) = crop_and_resize(imgmtx[i,:,:], width=resolution[0], height=resolution[1])
        else:
            (im,m,box) = crop_and_resize(imgmtx[i,:,:],mask=lblsmtx[i,:,:],width=resolution[0], height=resolution[1])
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
    img = ndimage.zoom(img,0.25,order=0)
    bodymask = img>maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3,3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask,iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int),connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions)>0:
        max_region = np.argmax(list(map(lambda x: x.area,regions)))+1
        bodymask = bodymask==max_region
        bodymask = ndimage.binary_dilation(bodymask,iterations=2)
    real_scaling = np.divide(oshape,img.shape)[0]
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(img,mask=None,width=192,height=192):
    bmask = simple_bodymask(img)
    img[bmask==0] = -1024
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg)>0:
        bbox = reg[0].bbox
    else:
        bbox = (0,0,bmask.shape[0],bmask.shape[1])
    img = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width,height])/np.asarray(img.shape),order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width,height])/np.asarray(mask.shape),order=0)
        #mask = ndimage.binary_closing(mask,iterations=5)
    return img,mask,bbox


def reshape_mask(mask,tbox,origsize):
    res = np.ones(origsize)*0
    resize=[tbox[2]-tbox[0], tbox[3]-tbox[1]]
    imgres = ndimage.zoom(mask,resize/np.asarray(mask.shape),order=0)

    res[tbox[0]:tbox[2],tbox[1]:tbox[3]] = imgres
    return res


class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx,None,:,:].astype(np.float)


def read_dicoms(path, primary=True, original=True):

    allfnames = []
    for dir,_,fnames in os.walk(path):
        [allfnames.append(os.path.join(dir,fname)) for fname in fnames]

    dcm_header_info = []
    dcm_parameters = []
    unique_set = [] #need this because too often there are duplicates of dicom files with different names
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
                            h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, dicom_header.ImagePositionPatient]
                            h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname, dicom_header.ImagePositionPatient]
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
    vol_unique = np.unique(conc, return_index=1, return_inverse=1) #unique volumes
    n_vol = len(vol_unique[1])
    logging.info('There are '+str(n_vol)+' volumes in the study')

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2]==curr_vol)[0]
        vol_files = dcm_header_info[info_idxs,2]
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs,3]])
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
        logging.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    else:
        logging.info(f'Looking for dicoms in {path}')
        dicom_vols = read_dicoms(path, original=False, primary=False)
        if len(dicom_vols) < 1:
            sys.exit('No dicoms found!')
        if len(dicom_vols) > 1:
            logging.warning("There are more than one volume in the path, will take the largest one")
        input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
    return input_image


