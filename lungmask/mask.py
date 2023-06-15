import logging
import os
import sys
import warnings
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
import skimage
import torch
from more_itertools import chunked
from tqdm import tqdm

from lungmask import utils

from .resunet import UNet

logging.basicConfig(
    stream=sys.stdout,
    format="lungmask %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

warnings.filterwarnings("ignore", category=UserWarning)


# stores urls and number of classes of the models
MODEL_URLS = {
    "R231": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth",
        3,
    ),
    "LTRCLobes": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth",
        6,
    ),
    "R231CovidWeb": (
        "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth",
        3,
    ),
}


def get_model(modelname: str, modelpath: Optional[str] = None) -> torch.nn.Module:
    """Loads specific model and state

    Args:
        modelname (str): Modelname (e.g. R231, LTRCLobes or R231CovidWeb)
        modelpath (Optional[str], optional): Path to statedict, if not provided will be downloaded automatically. Modelname will be ignored if provided. Defaults to None.

    Returns:
        torch.nn.Module: Loaded model in eval state
    """
    if modelpath is None:
        model_url, n_classes = MODEL_URLS[modelname]
        state_dict = torch.hub.load_state_dict_from_url(
            model_url, progress=True, map_location=torch.device("cpu")
        )
    else:
        state_dict = torch.load(modelpath, map_location=torch.device("cpu"))

    n_classes = len(list(state_dict.values())[-1])

    model = UNet(
        n_classes=n_classes,
        padding=True,
        depth=5,
        up_mode="upsample",
        batch_norm=True,
        residual=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


class LMInferer:
    def __init__(
        self,
        modelname: str = "R231",
        modelpath: Optional[str] = None,
        fillmodel: Optional[str] = None,
        fillmodel_path: Optional[str] = None,
        force_cpu: bool = False,
        batch_size: int = 20,
        volume_postprocessing: bool = True,
        noHU: bool = False,
        tqdm_disable: bool = False,
    ):
        """LungMaskInference

        Args:
            modelname (str, optional): Model to be applied. Defaults to 'R231'.
            modelpath (str, optional): Path to modeleights. `modelname` parameter will be ignored if provided. Defaults to None.
            fillmodel (Optional[str], optional): Fillmodel to be applied. Defaults to None.
            fillmodel_path (Optional[str], optional): Path to weights for fillmodel. `fillmodel` parameter will be ignored if provided. Defaults to None.
            force_cpu (bool, optional): Will not use GPU is `True`. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 20.
            volume_postprocessing (bool, optional): If `Fales` will not perform postprocessing (connected component analysis). Defaults to True.
            noHU (bool, optional): If `True` no HU intensities are expected. Not recommended. Defaults to False.
            tqdm_disable (bool, optional): If `True`, will disable progress bar. Defaults to False.
        """
        assert (
            modelname in MODEL_URLS
        ), "Modelname not found. Please choose from: {}".format(MODEL_URLS.keys())
        if fillmodel is not None:
            assert (
                fillmodel in MODEL_URLS
            ), "Modelname not found. Please choose from: {}".format(MODEL_URLS.keys())

        # if paths provided, overwrite name
        if modelpath is not None:
            modelname = os.path.basename(modelpath)
        if fillmodel_path is not None:
            fillmodel = os.path.basename(fillmodel_path)

        self.fillmodel = fillmodel
        self.modelname = modelname
        self.force_cpu = force_cpu
        self.batch_size = batch_size
        self.volume_postprocessing = volume_postprocessing
        self.noHU = noHU
        self.tqdm_disable = tqdm_disable

        self.model = get_model(self.modelname, modelpath)

        self.device = torch.device("cpu")
        if not self.force_cpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                logging.info("No GPU found, using CPU instead")
        self.model.to(self.device)

        self.fillmodelm = None
        if self.fillmodel is not None:
            self.fillmodelm = get_model(self.fillmodel, fillmodel_path)
            self.fillmodelm.to(self.device)

    def _inference(
        self, image: Union[sitk.Image, np.ndarray], model: torch.nn.Module
    ) -> np.ndarray:
        """Performs model inference

        Args:
            image (Union[sitk.Image, np.ndarray]): Input image (volumetric)
            model (torch.nn.Module): Model to be applied

        Returns:
            np.ndarray: Inference result
        """
        numpy_mode = isinstance(image, np.ndarray)
        if numpy_mode:
            inimg_raw = image.copy()
        else:
            curr_orient = (
                sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                    image.GetDirection()
                )
            )
            if curr_orient != "LPS":
                image = sitk.DICOMOrient(image, "LPS")
            inimg_raw = sitk.GetArrayFromImage(image)

        if self.noHU:
            # support for non HU images. This is just a hack. The models were not trained with this in mind
            tvolslices = skimage.color.rgb2gray(inimg_raw)
            tvolslices = skimage.transform.resize(tvolslices, [256, 256])
            tvolslices = np.asarray([tvolslices * x for x in np.linspace(0.3, 2, 20)])
            tvolslices[tvolslices > 1] = 1
            sanity = [
                (tvolslices[x] > 0.6).sum() > 25000 for x in range(len(tvolslices))
            ]
            tvolslices = tvolslices[sanity]
        else:
            tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
            tvolslices[tvolslices > 600] = 600
            tvolslices = np.divide((tvolslices + 1024), 1624)

        timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

        with torch.no_grad():
            for mbnp in tqdm(
                chunked(tvolslices, self.batch_size),
                disable=self.tqdm_disable,
                total=len(tvolslices) / self.batch_size,
            ):
                mbt = torch.as_tensor(
                    np.asarray(mbnp)[:, None, ::],
                    dtype=torch.float32,
                    device=self.device,
                )
                prediction = model(mbt)
                pred = (
                    torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
                )
                timage_res = np.vstack((timage_res, pred))

        # postprocessing includes removal of small connected components, hole filling and mapping of small components to
        # neighbors
        if self.volume_postprocessing:
            outmask = utils.postprocessing(timage_res, disable_tqdm=self.tqdm_disable)
        else:
            outmask = timage_res

        if self.noHU:
            outmask = skimage.transform.resize(
                outmask[np.argmax((outmask == 1).sum(axis=(1, 2)))],
                inimg_raw.shape[:2],
                order=0,
                anti_aliasing=False,
                preserve_range=True,
            )[None, :, :]
        else:
            outmask = np.asarray(
                [
                    utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:])
                    for i in range(outmask.shape[0])
                ],
                dtype=np.uint8,
            )

        if not numpy_mode:
            if curr_orient != "LPS":
                outmask = sitk.GetImageFromArray(outmask)
                outmask = sitk.DICOMOrient(outmask, curr_orient)
                outmask = sitk.GetArrayFromImage(outmask)

        return outmask.astype(np.uint8)

    def apply(self, image: Union[sitk.Image, np.ndarray]) -> np.ndarray:
        """Apply model on image (volumetric)

        Args:
            image (Union[sitk.Image, np.ndarray]): Input image

        Returns:
            np.ndarray: Lung segmentation
        """
        if self.fillmodel is None:
            return self._inference(image, self.model)
        else:
            logging.info(f"Apply: {self.modelname}")
            res_l = self._inference(image, self.model)
            logging.info(f"Apply: {self.fillmodel}")
            res_r = self._inference(image, self.fillmodelm)
            spare_value = res_l.max() + 1
            res_l[np.logical_and(res_l == 0, res_r > 0)] = spare_value
            res_l[res_r == 0] = 0
            logging.info("Fusing results... this may take up to several minutes!")
            return utils.postprocessing(res_l, spare=[spare_value])


def apply(
    image: Union[sitk.Image, np.ndarray],
    model=None,
    force_cpu=False,
    batch_size=20,
    volume_postprocessing=True,
    noHU=False,
    tqdm_disable=False,
):
    warnings.warn(
        "The function `apply` will be removed in a future version. Please use the LMInferer class!",
        DeprecationWarning,
    )
    inferer = LMInferer(
        force_cpu=force_cpu,
        batch_size=batch_size,
        volume_postprocessing=volume_postprocessing,
        noHU=noHU,
        tqdm_disable=tqdm_disable,
    )
    if model is not None:
        inferer.model = model.to(inferer.device)
    return inferer.apply(image)


def apply_fused(
    image,
    basemodel="LTRCLobes",
    fillmodel="R231",
    force_cpu=False,
    batch_size=20,
    volume_postprocessing=True,
    noHU=False,
    tqdm_disable=False,
):
    warnings.warn(
        "The function `apply_fused` will be removed in a future version. Please use the LMInferer class!",
        DeprecationWarning,
    )
    inferer = LMInferer(
        modelname=basemodel,
        force_cpu=force_cpu,
        fillmodel=fillmodel,
        batch_size=batch_size,
        volume_postprocessing=volume_postprocessing,
        noHU=noHU,
        tqdm_disable=tqdm_disable,
    )
    return inferer.apply(image)
