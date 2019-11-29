import numpy as np
import torch
from . import utils
import SimpleITK as sitk
import skimage
from .resunet import UNet
import scipy.ndimage as ndimage
import warnings
import sys
from tqdm import tqdm

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore",category=UserWarning)


model_urls = {('unet','R231'): 'http://www.cir.meduniwien.ac.at/downloads/unet_r231-28d0c9ef.pth'}


def apply(image, model, force_cpu=False, batch_size=20, volume_postprocessing=True, show_process=True):

    voxvol = np.prod(image.GetSpacing())
    inimg_raw = sitk.GetArrayFromImage(image)
    del image

    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.info("No GPU support available, will use CPU. Note, that this is significantely slower!")
            batch_size = 1
            device = torch.device('cpu')
    model.to(device)


    tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
    tvolslices[tvolslices > 600] = 600
    tvolslices = np.divide((tvolslices+1024),1624)
    torch_ds_val = utils.LungLabelsDS_inf(tvolslices)
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    timage_res = np.empty((np.append(0,tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        for X in tqdm(dataloader_val):
            X = X.float().to(device)
            prediction = model(X)
            pls = torch.max(prediction,1)[1].detach().cpu().numpy().astype(np.uint8)
            timage_res = np.vstack((timage_res, pls))

    if volume_postprocessing:
        area = 25000/voxvol
        regionmask = skimage.measure.label(np.asarray(timage_res).astype(np.int32))
        regions = skimage.measure.regionprops(regionmask)
        resizes = np.asarray([x.area for x in regions])
        m = len(resizes)
        ix = np.zeros((m,), dtype=np.uint8)
        ix[resizes > area] = 1
        ix = np.concatenate([[0, ], ix])
        outmask = ix[regionmask]

        outmaskr = (timage_res==1) & (outmask>0)
        outmaskl = (timage_res==2) & (outmask>0)
        outmaskr = ndimage.binary_fill_holes(outmaskr)
        outmaskl = ndimage.binary_fill_holes(outmaskl)
        outmask[outmaskl] = 2
        outmask[outmaskr] = 1
        outmask = outmask.astype(np.uint8)
    else:
        outmask = timage_res

    outmask = np.asarray([utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])], dtype=np.uint8)

    return outmask


def get_model(modeltype, modelname):
    model_url = model_urls[(modeltype, modelname)]
    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))
    if modeltype == 'unet':
        model = UNet(n_classes=3, padding=True,  depth=5, up_mode='upsample', batch_norm=True, residual=False)
    elif modeltype == 'resunet':
        model = UNet(n_classes=3, padding=True,  depth=5, up_mode='upsample', batch_norm=True, residual=True)
    else:
        logging.exception(f"Model {modelname} not known")
    model.load_state_dict(state_dict)
    model.eval()
    return model

