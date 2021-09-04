import numpy as np
import torch
#from lungmask import utils
import utils
import SimpleITK as sitk
#from .resunet import UNet
from resunet import UNet
import warnings
import sys
from tqdm import tqdm
import skimage
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# stores urls and number of classes of the models
model_urls = {('unet', 'R231'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth', 3),
              ('unet', 'LTRCLobes'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth', 6),
              ('unet', 'R231CovidWeb'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth', 3),
              ('unet', 'R231lung1'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231lung1-1eab9955.pth', 3)}


def apply(image, model=None, force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
    if model is None:
        model = get_model('unet', 'R231lung1')
    
    #consiguiendo pixel_array de la imagen  (sea por dicom o nii)
    #ojo cambia el orden de las dimensiones de (1024*1024*100) a (100*1024*1024)
    inimg_raw = sitk.GetArrayFromImage(image)
    
    directions = np.asarray(image.GetDirection())
    
    if len(directions) == 9:
        inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0])

    #liberar de la memoria la imagen simpleitk y solo quedarse con su array
    del image

    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            device = torch.device('cpu')
    
    #pasando modelo a gpu
    model.to(device)

    #----------------------------------- PRE-PROCESAMIENTO -----------------------------------#
    
    if not noHU:
        #pixeles de todos los cortes preprocesados
        #return [ imagen de la region redimensionada, (xnew_box) mascara de la region redimensionada, bbox de region]
        #se puede pasarle un label pero esto asumo que se aplica en caso de entrenamiento
        tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
        
        #umbralizando
        tvolslices[tvolslices > 600] = 600
        #normalizarlo 1624 = 1024 + 600
        tvolslices = np.divide((tvolslices + 1024), 1624)
    else:
        # support for non HU images. This is just a hack. The models were not trained with this in mind
        #pixeles de todos los cortes jpg/png to gray
        tvolslices = skimage.color.rgb2gray(inimg_raw)

        #escalar a imagen de 256 x 256
        tvolslices = skimage.transform.resize(tvolslices, [256, 256])
        #varias convoluciones de cada valor en el vector  [0.3, 2] de 20 valores
        tvolslices = np.asarray([tvolslices*x for x in np.linspace(0.3,2,20)])
        #despues de la normalizacion aplicar umbral de 1
        tvolslices[tvolslices>1] = 1
        #el nro de pixels mayor a 0.6 debe ser mas grande q 25000 (buena cantidad)
        sanity = [(tvolslices[x]>0.6).sum()>25000 for x in range(len(tvolslices))]
        #restringir los pixels a la condicion anterior 
        tvolslices = tvolslices[sanity]
    
    #----------------------------------------------------------------------------------------#
    #----------------------------------- INIT DATA-LOADER -----------------------------------#

    #enviar todos los cortes (formato pixel) al Dataset y pasarlo dataloader
    torch_ds_val = utils.LungLabelsDS_inf(tvolslices)
    #enviar al dataloader
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=False)
    
    #preparando almacen del resultado
    #np.append agregar dimension y lo concatena con cero [0,512] solo para armar un array
    timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)
    
    #----------------------------------------------------------------------------------------#
    #--------------------------------------- PREDICTION -------------------------------------#

    with torch.no_grad():
        for X in tqdm(dataloader_val):
            X = X.float().to(device)
            prediction = model(X)
            #el tensor tuviera valores por cada columna con las distintas mascaras de diferente probabildidad
            #se elige la mayor de ellas (elegir una mascara con la mas alta probabilidad)
            #pls son los indices de estos valores maximos del tensor
            pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
            #al ser indices los apila al resultado no inicializado
            timage_res = np.vstack((timage_res, pls))
    
    #----------------------------------------------------------------------------------------#
    #--------------------------------- POST-PROCESAMIENTO ------------------------------------#

    # postprocessing includes removal of small connected components, hole filling and mapping of small components to
    # neighbors
    if volume_postprocessing:
        #aplicar post-procesamiento
        outmask = utils.postrocessing(timage_res)
    else:
        outmask = timage_res

    if noHU:
        #aplicar redimension con tnrasform en caso de jpg/png
        outmask = skimage.transform.resize(outmask[np.argmax((outmask==1).sum(axis=(1,2)))], inimg_raw.shape[:2], order=0, anti_aliasing=False, preserve_range=True)[None,:,:]
    else:
        #en caso de imagen medica usamos el xnew_box para la nueva redimension
         outmask = np.asarray(
            [utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
            dtype=np.uint8)
    
    #un giro para corregir la orientacion
    if len(directions) == 9:
        outmask = np.flip(outmask, np.where(directions[[0,4,8]][::-1]<0)[0])

    return outmask.astype(np.uint8)
    #-----------------------------------------------------------------------------------------#


def get_model(modeltype, modelname):
    model_url, n_classes = model_urls[(modeltype, modelname)]
    print("modelo: {}".format(model_url))
    print("classes: {}".format(n_classes))
    #state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))

    state_dict = torch.load("unet_r231lung1-1eab9955.pth")


    if modeltype == 'unet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
    elif modeltype == 'resunet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=True)
    else:
        logging.exception(f"Model {modelname} not known")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_fused(image, basemodel = 'LTRCLobes', fillmodel = 'R231', force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
    '''Will apply basemodel and use fillmodel to mitiage false negatives'''
    mdl_r = get_model('unet',fillmodel)
    mdl_l = get_model('unet',basemodel)
    
    logging.info("Apply: %s" % basemodel)
    res_l = apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    
    logging.info("Apply: %s" % fillmodel)
    res_r = apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    
    spare_value = res_l.max()+1
    #usar res_r para umbralizar cuando sea mayor(max res_l) e igual a cero (cero)
    res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
    res_l[res_r==0] = 0
    logging.info("Fusing results... this may take up to several minutes!")
    #aplicar tambien postprocesamiento sobre el nuevo res_l
    return utils.postrocessing(res_l, spare=[spare_value])
