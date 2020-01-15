Automated lung segmentation in CT under presence of severe pathologies
================================================================================

This package applies a pre-trained U-net model for lung segmentation. 

Description and evaluation of the model can be found here:
* Hofmanninger et al., ---

The model performs segmentation on individual slices, extracts right-left lung seperately includes airpockets, tumors and effusions. The trachea is not included in the lung.

Installation
------------
```
pip install git+https://github.com/JoHof/lungmask
```

Usage
-----
As a command line tool:
```
lungmask.py [inputpath] [outputpath]
```
If inputpath points to a file, the file will be loaded. If input path points to a directory, the directory is searched for DICOM series. The largest volume found (in terms of number of voxels) will be used to compute the lungmask. It is recommended to provide a directory with a single series.

Type:
```
lungmask.py -h
```
for additional options.

as a python module:

 