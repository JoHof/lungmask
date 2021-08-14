import os
import sitk
import logging



class InputImage:
    """
    The Target defines the domain-specific interface used by the client code.
    """

    def __init__(self, path):
        if os.path.isfile(path):
            logging.info(f'Read input: {path}')
            self.input_image = sitk.ReadImage(path)
        else:
            self.input_image = None

    def load(self):
        return self.input_image

class DicomsImage:
    """
    The Adaptee contains some useful behavior, but its interface is incompatible
    with the existing client code. The Adaptee needs some adaptation before the
    client code can use it.
    """

    def __init__(self, path):
        logging.info(f'Looking for dicoms in {path}')
        self.input_image = self._read_dicoms(path, original=False, primary=False)

    def load(self):
        return self.input_image

    def _read_dicoms(path, primary=True, original=True):
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

class Adapter(InputImage):
    """
    The Adapter makes the Adaptee's interface compatible with the Target's
    interface via composition.
    """
    def __init__(self, adaptee: DicomsImage):
        self.adaptee = adaptee

    def load(self):
        dicom_vols = self.adaptee.load()
        if len(dicom_vols) < 1:
            sys.exit('No dicoms found!')
        if len(dicom_vols) > 1:
            logging.warning("There are more than one volume in the path, will take the largest one")
        input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
        return input_image

def get_input_image(path):
    if os.path.isfile(path):
        input_image = InputImage(path).load()
    else:
        dicoms_image = DicomsImage(path)
        adapter = Adapter(dicoms_image)
        input_image = adapter.load()
    return input_image



