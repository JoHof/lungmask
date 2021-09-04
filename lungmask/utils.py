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


#----------------------------------- PRE-PROCESAMIENTO -----------------------------------#

def preprocess(img, label=None, resolution=[192, 192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)
    
    #restringir HU
    #umbralizar valores mas de -1024
    imgmtx[imgmtx < -1024] = -1024
    #umbralizar valores mas de 600
    imgmtx[imgmtx > 600] = 600
    
    #cip (compute image preprocessing)
    cip_xnew = []
    cip_box = []
    cip_mask = []
    
    #para cada corte de la imagen
    for i in range(imgmtx.shape[0]):
        #si no hay etiqueta o mascara
        if label is None:

            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0], height=resolution[1])
            cip_mask.append(m)
        
        cip_xnew.append(im)
        cip_box.append(box)

    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)


#identificacion de la region mas grande y preprocesada (zooming,closing,fillholes,erosion,dilation)
def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    #128/dimImg es el factor del zoom a aplicar
    img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
    #mascara del cuerpo cercano al pulmon
    bodymask = img > maskthreshold
    #erosion of the dilation of the image
    bodymask = ndimage.binary_closing(bodymask)
    #Fill the holes in binary object
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    #apply erosion to binary object
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    #retorna array de enteros que identifican a una particular ragion conectada
    #las regiones conectadas agrupan pixels vecinos del mismo valor
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    #propiedades de cada region dado pro label bodymask
    regions = skimage.measure.regionprops(bodymask.astype(int))
    #si encontro regiones
    if len(regions) > 0:
        #id de region de mayor area + 1 (id == intensidad pixel)?
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1

        #identificar maxima region
        bodymask = bodymask == max_region
        #aplicar dilation
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    
    #recuperar del zooming
    real_scaling = np.asarray(oshape)/128
    return ndimage.zoom(bodymask, real_scaling, order=0)


#CROP por reducir a un area especifica de un region identificada del pulmon
#RESIZE por redimensionar a un tamanio de 256 por 256 que es lo que Imagenet trabaja
#return [ imagen de la region redimensionada, mascara de la region redimensionada, bbox de region]
def crop_and_resize(img, mask=None, width=192, height=192):

    #identificar la region predilecta (mas grande y que sea del pulmon)
    bmask = simple_bodymask(img)
    
    # asumo que puede eliminar regiones del pulmon si asignamos de maneral general -1024 a todas    #las zonas que no son pulmones (podrian estar dentro del pulmon)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help


    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    #captura el bbox que encierra la region identificada si hubiera otro
    #si no lo hay crea un bbox del anterior mascara dada por simple_bodymask
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])

    #capturar la imagen de la bbox de la region (se puede obtener por propiedad ToDo)
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    #hacer un zoom de 256*256 con respecto al tamanio de ingreso (resize)
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    
    #En caso de contar con una mascara 
    if not mask is None:
        #aplicar igualmente el ROI o bbbox sobre la mascara
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        #y tambien aplicar el resize de 256*256
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


## For some reasons skimage.transform leads to edgy mask borders compared to ndimage.zoom
# def reshape_mask(mask, tbox, origsize):
#     res = np.ones(origsize) * 0
#     resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
#     imgres = skimage.transform.resize(mask, resize, order=0, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)
#     res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
#     return res


#Redimensionar y Limpiar la Mascara de Salida
def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    #nueva dimension
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    #aplicar cambion de resolucion
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    #el resto de pixeles seran llenados de ceros
    #por lo cual sigue teniendo la misma cantidad de pixels
    #pero con borde negro y al interior la imagen redimensionada
    res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
    return res

#----------------------------------------------------------------------------------------#
#----------------------------------- INIT DATA-LOADER -----------------------------------#

#Dataset de inferencia (construcotr, largo y conseguir item)
class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #borra la dimension del corte
        return self.dataset[idx, None, :, :].astype(np.float)

#----------------------------------------------------------------------------------------#
#----------------------------------- LECTURA DE IMAGENES  -----------------------------------#

def read_dicoms(path, primary=True, original=True):
    
    #lee de todos los directorios de dicom que existan en un directorio
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    dcm_parameters = []
    
    # need this because too often there are duplicates of dicom files with different names
    unique_set = [] 
    i = 0
    
    #recorriendo cada corte
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1

        #lectura de cortes de los distintos directorios
        if filename_[0] != 'DICOMDIR':
            try:
                #defer size en caso la ram no de para leer todo
                #se lee la imagen pra cuando se lea, cada 100mb
                #stop_before_pixels: se detiene justo antes de la lectura de pixeles
                #y no sobrecargar la memoria
                #force: leer incluso si el corte no tiene metadata (False default)
                dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                

                #en caso de que exista metadata del dicom (todo va bien)
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
                            #[id estudio, id serie, posicionPaciente]
                            h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, dicom_header.ImagePositionPatient]

                            #[id_estudio, idSerie, path, posicionPaciente]
                            h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname, dicom_header.ImagePositionPatient]

                            #si existiera otro estudio,serie,posicion en otra dicom
                            #no se considera dentro del unique_set
                            if h_info_wo_name not in unique_set:

                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)

                                #agregar parametros importantes de las dicoms
                                # kvp = None
                                # if 'KVP' in dicom_header:
                                #     kvp = dicom_header.KVP
                                # dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)
    
    #conjunto de series
    conc = [x[1] for x in dcm_header_info]
    #lista de los ids (posicion) ordenadas de las series
    sidx = np.argsort(conc)
    #nuevo conjunto de series ordenado
    conc = np.asarray(conc)[sidx]
    #nuevo conjunto de metadata (path) ordenada
    dcm_header_info = np.asarray(dcm_header_info)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    
    #conjunto de id series ordenados y unicos (ids)
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes

    #vol_unique tiene: [ valores ordenados series, indices_original, idx_inversa_from_uniques]
    
    n_vol = len(vol_unique[1])
    #hay n series en todos los directorios dicom
    logging.info('There are ' + str(n_vol) + ' volumes in the study')

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i

        #devuelve el id dentro del vol_unique que coincide con el curr_vol
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        
        #ruta del actual corte (2) de los ids dados en info_idxs
        vol_files = dcm_header_info[info_idxs, 2]
        
        #posicion del paciente (valor del corner 1er pixel de lectura)
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
        
        #ordenar series a partir de las posiciones
        slicesort_idx = np.argsort(positions)

        #rutas ordenadas de los distintas series que posean curr_vol
        vol_files = vol_files[slicesort_idx]
        
        #se agrega rutas de las series con el actual curr_vol
        relevant_series.append(vol_files)

        #leer las distintas series de la imagen dicom
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        
        #se agrega el set de volumenes(distintas series) al la lista de volumenes
        relevant_volumes.append(vol)

    return relevant_volumes, relevant_series


def get_input_image(path):
    if os.path.isfile(path):
        #probando leer una unica image
        logging.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    else:
        #probando leer varias imagenes
        logging.info(f'Looking for dicoms in {path}')
        
        #lectura de dicoms
        dicom_vols,paths_series = read_dicoms(path, original=False, primary=False)
        
        if len(dicom_vols) < 1:
            sys.exit('No dicoms found!')
        if len(dicom_vols) > 1:
            #tomando la dicom de mayor nro de cortes
            logging.warning("There are more than one volume in the path, will take the largest one")
        #consiguiendo el id de la dicom que tiene mas pixels/voxels
        idmax = np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)
        input_image = dicom_vols[idmax]
        path_image = paths_series[idmax]
    return input_image, path_image




#------------------------------------------------------------------------------------------#
#------------------------------------ POST - PROCESSING -----------------------------------#

#procesa la mascara de salida (label)
def postrocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''
    #la salida no solo tiene una solo label, en conjunto la final prediccion
    #cuenta con diferentes labels, algunos seran unidos por vecindad

    # merge small components to neighbours
    
    # identificar las regiones del label_image
    regionmask = skimage.measure.label(label_image)
    # quedarse con los label unicos
    origlabels = np.unique(label_image)

    #mas largo componente conectado (region) (vacio)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    
    #propiedades de las regiones del regionmask (label_image es dado como intensidad de imagen)
    regions = skimage.measure.regionprops(regionmask, label_image)
    #ordenar por areas
    regions.sort(key=lambda x: x.area)

    #de cada region obtenemos su respectivo label (con la intensidad de imagen)
    #obtenemos el label (The label in the labeled input image)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        if r.area > origlabels_maxsub[r.max_intensity]:
            #el nueva area mas larga sera r.area
            origlabels_maxsub[r.max_intensity] = r.area
            #en cada regionlabel identificada con r.label sera puesto la
            #maxima intensidad de dicha region que alberga ese (label de region)
            region_to_lobemap[r.label] = r.max_intensity
    
    #recorremos cada region (sabremos cuantas regiones hay en la mascara de salida)
    for r in tqdm(regions):
        #para el caso de pequeñas regiones 
        if (r.area < origlabels_maxsub[r.max_intensity] or r.max_intensity in spare) and r.area>2: # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            
            bb = bbox_3D(regionmask == r.label)
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
            #asignado con el valor n que es la mas larga subregion
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            

            
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                regions[regionlabels.index(mapto)].max_intensity]:
                
                #actualizar el area con myarea(acumulado) de la subregion mas grande
                origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
            
            #
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea
    
    #label de region a original label(outmask_mapped)
    #mapeado con el padre de todas las intensidades la mascara mapeada
    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped==spare] = 0 

    if outmask_mapped.shape[0] == 1: #binario
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver

        #area_closing in binary case is  remove_small_holes
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        #en caso de no ser una mascara binaria (ademas tiene grises)
        holefiller = fill_voids.fill
    
    #esqueleto del resultado con la dimension de la mascara ya aislada
    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)

    #mantenemos los valores unicos de las mascaras aisladas (mas grandes)
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
    
    if rmax-rmin == 0:
        rmax = rmin+1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask
