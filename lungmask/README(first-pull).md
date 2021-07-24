Aggregate application 1
We created a new class named ImageDataHandler to encapsulate activities within image data IO and processing domain. 
Class diagram is as below. 


Images and tables are shown in report(PDF) 
 
Changes made compare to original code base are listed as below. 
File	Class	Comment
data.py	ImageDataHandler	A new class that represents the aggregate for image IO and pre or post-processing.
data.py	ImageLoader	A new class that encapsulates data loading activities.  
data.py	ImagePreProcessor	A new class related to data pre-processing activities. 
data.py	ImagePostProcessor	A new class related to data post-processing activities. 
data.py	ImageSaver	A new class that encapsulates data saving activities. 

More details on the class design are discussed below. 
1.	ImageDataHandler is a new class designed as aggregate root. All image data activities are only accessible via ImageDataHandler. It consists four entities, of which two are related to image file IO (ImageLoader and ImageSaver) and two are related to image processing (ImagePreProcessor and ImagePostProcessor). It refers to two attributes (‘in_file’ and ‘out_file’) from user parameters represented as an object ‘args’ from domain ‘argparse’. It is the only accessible point to image data activities. 
2.	ImageLoader represents image loading functions. It deals with the logic to process different image file formats internally given the input file location. Functions in the original project including ‘get_input_image’ and ‘read_dicoms’ in utils.py are moved into ImageLoader. The function ‘read_dicoms’ is set as private and renamed as ‘_read_dicoms’. It currently is only used by the function ‘get_input_image’. 
3.	ImageSaver saves file to designated file location specified by output file argument in ImageDataHandler. The functionality is originally implemented in function ‘main’ in ‘__main__.py’. It is now part of ImageDataHandler aggregate. This makes it reusable by other parts of program and we can add more business logic in the future if necessary.
4.	ImagePreProcessor handles data pre-processing tasks prior to applying mask to images. Functions ‘preprocess’, ‘crop_and_resize’ and ‘simple_bodymask’ in ‘utils.py’ are grouped into this domain. Functions ‘crop_and_resize’ and ‘simple_bodymask’ are set as private and renamed. They are currently only used by the function ‘preprocess’’. 
5.	ImagePostProcessor handles tasks after applying mask to images. Functions ‘postprocessing’, ‘bbox_3D’ and ‘keep_largest_connected_component’ in ‘utils.py’ are grouped into this domain. Functions ‘‘bbox_3D’ and ‘keep_largest_connected_component’ are set as private and renamed. They are currently only used by the function ‘postprocess’’. 

In the original project, all data related functions are grouped in utils.py and can be publicly accessed by any outside function. By using a domain model, we create a boundary within the aggregate class ImageDataHandler. It represents the lifecycle of an image data object within the application, i.e. loaded from file system -> processed prior to mask -> processed after mask -> saved to file system. The sequential flow of business logic makes it easier to modify, add and test. Several functions only serve as utility to specific features, for example, function ‘crop_and_resize’, ‘simple_bodymask’, ‘bbox_3D’, ‘read_dicoms’. They are now assigned to new domain classes and hidden from outside. This will prevent misusage by other parts of program. Also, the single access point at aggregate root ensures the input and output params are consistently used within the data domain.
