import numpy as np
import torch
import SimpleITK as sitk
from .resunet import UNet
import warnings
import sys
from tqdm import tqdm
import skimage
import logging
from lungmask.data import ImageDataHandler

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :].astype(np.float)


class MaskHandler:
    def __init__(self, args):

        # arguments
        self.args = args
        self.batchsize = args.batchsize
        if args.cpu:
            self.batchsize = 1

        # use ImageDataHandler to handle actions related to image data processing
        self.datahandle = ImageDataHandler(args.input, args.output)
        # load image for processing
        self.image = self.datahandle.get_input_image()

        self.result = None

    def apply_masks(self):
        """apply masks according to user arguments"""
        if self.args.modelname == 'LTRCLobes_R231':
            self.result = self._apply_fused()
        else:
            model = self._get_model(self.args.modeltype, self.args.modelname)
            self.result = self._apply(model)

        if args.noHU:
            file_ending = args.output.split('.')[-1]
            print(file_ending)
            if file_ending in ['jpg', 'jpeg', 'png']:
                self.result = (self.result / (self.result.max()) * 255).astype(np.uint8)
            self.result = self.result[0]
        return self.result

    def save_results(self):
        self.datahandle.save_results(self.result, self.image)

    def _reshape_mask(self, mask, tbox, origsize):
        """function used only in function self._apply. set as private"""
        res = np.ones(origsize) * 0
        resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
        imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
        res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
        return res

    def _apply(self, model=None):
        """function used only in function self.apply_masks. set as private"""

        force_cpu = self.args.cpu
        batch_size = self.batchsize
        volume_postprocessing = not (self.args.nopostprocess)
        noHU = self.args.noHU

        if model is None:
            model = self._get_model('unet', 'R231')

        numpy_mode = isinstance(self.image, np.ndarray)
        if numpy_mode:
            inimg_raw = self.image.copy()
        else:
            inimg_raw = sitk.GetArrayFromImage(self.image)
            directions = np.asarray(self.image.GetDirection())
            if len(directions) == 9:
                inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0])

        if force_cpu:
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
                batch_size = 1
                device = torch.device('cpu')
        model.to(device)


        if not noHU:
            tvolslices, xnew_box = self.datahandle.preprocess(inimg_raw, resolution=[256, 256])
            tvolslices[tvolslices > 600] = 600
            tvolslices = np.divide((tvolslices + 1024), 1624)
        else:
            # support for non HU images. This is just a hack. The models were not trained with this in mind
            tvolslices = skimage.color.rgb2gray(inimg_raw)
            tvolslices = skimage.transform.resize(tvolslices, [256, 256])
            tvolslices = np.asarray([tvolslices*x for x in np.linspace(0.3,2,20)])
            tvolslices[tvolslices>1] = 1
            sanity = [(tvolslices[x]>0.6).sum()>25000 for x in range(len(tvolslices))]
            tvolslices = tvolslices[sanity]
        torch_ds_val = LungLabelsDS_inf(tvolslices)
        dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1,
                                                     pin_memory=False)

        timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

        with torch.no_grad():
            for X in tqdm(dataloader_val):
                X = X.float().to(device)
                prediction = model(X)
                pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
                timage_res = np.vstack((timage_res, pls))

        # postprocessing includes removal of small connected components, hole filling and mapping of small components to
        # neighbors
        if volume_postprocessing:
            outmask = self.datahandle.postprocessing(timage_res)
        else:
            outmask = timage_res

        if noHU:
            outmask = skimage.transform.resize(outmask[np.argmax((outmask==1).sum(axis=(1,2)))], inimg_raw.shape[:2], order=0, anti_aliasing=False, preserve_range=True)[None,:,:]
        else:
             outmask = np.asarray(
                [self._reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
                dtype=np.uint8)

        if not numpy_mode:
            if len(directions) == 9:
                outmask = np.flip(outmask, np.where(directions[[0,4,8]][::-1]<0)[0])

        return outmask.astype(np.uint8)


    def _get_model(self, modeltype, modelname):
        """function used only in function self._apply_masks and self._apply_fused. set as private."""
        model_url, n_classes = self.model_urls[(modeltype, modelname)]
        state_dict = torch.hub.load_state_dict_from_url(self.model_url, progress=True, map_location=torch.device('cpu'))
        if modeltype == 'unet':
            model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
        elif modeltype == 'resunet':
            model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=True)
        else:
            logging.exception(f"Model {modelname} not known")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _apply_fused(self, basemodel = 'LTRCLobes', fillmodel = 'R231'):
        """function used only in function self.apply_masks. set as private"""
        '''Will apply basemodel and use fillmodel to mitiage false negatives'''

        mdl_r = self._get_model('unet',fillmodel)
        mdl_l = self._get_model('unet',basemodel)
        logging.info("Apply: %s" % basemodel)
        res_l = self._apply(self.image, mdl_l)
        logging.info("Apply: %s" % fillmodel)
        res_r = self._apply(self.image, mdl_r)
        spare_value = res_l.max()+1
        res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
        res_l[res_r==0] = 0
        logging.info("Fusing results... this may take up to several minutes!")
        return self.datahandle.postprocessing(res_l, spare=[spare_value])
