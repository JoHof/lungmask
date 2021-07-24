Aggregate application 2
A new class MaskHandler is created to represent the whole image mask process. The relationship between MaskHandler and other domains are shown below.

Images and tables are shown in report(PDF)  
 
Changes made are detailed as follows.
1.	The aggregate root MaskHandler is the only accessible point to carry out the image mask task. Changes are made in the function ‘main()’ in ‘__main__.py’ to remove logics on input/output flow and conditional mask activities in the original file. Instead a ‘maskhandle’ object is created and called to perform required functions. In the aggregate, the image masking task is delegated to the function ‘apply_masks’ in MaskHandler. Task to save processed images is done by ‘save_results’ in MaskHandler. 
2.	MaskHandler represents a wider domain that includes data processing, neural network model loading, and image masking tasks. ImageDataHandler and Unet are entities within the aggregate MaskHandler. It ensures the same data processing logic and neural network model logic for all mask activities.
3.	The class ‘LungLabelsDS_inf’ is moved from ‘utils.py’ to ‘mask.py’. 
4.	Functions ‘get_model’, ‘apply’, ‘apply_fused’ in ‘mask.py’ are renamed as private functions in MaskHandler. They handle detailed logic in data and neural network model which are only relevant to domain experts. They are only used by function ‘apply_masks’ in MaskHandler. 
5.	The function ‘reshape_mask’ in ‘utils.py’ is moved under MaskHandler, renamed and set as private function. It is only used by function ‘_apply’ in MaskHandler.
6.	User command-line arguments, the image data and the final result are referenced consistently in MaskHandler. For example, user command-line arguments are referenced as ‘self.args’ in MaskHandler. ImageDataHandler can only refer to ‘self.args.input’ and ‘self.args.output’ for data input and output locations. The image data is represented as an attribuate ‘self.image’. Functions ‘self._apply()’ and ‘self._apply_fused()’ refer to ‘self.image’ as the image file for masking task. No alteration is done to ‘self.image’ in MaskHandler. The process image is stored in the class attribute ‘self.result’. The function ‘self.apply_mask()’ stores masked image into ‘self.result’ and function ‘self.save_results’ makes ‘self.result’ persistent in file system. 

MaskHandler groups entities ImageDataHandler and Unet and seperates business logic from the main in the original project. The logics such as image files IO, image formatting and neural networks forward calculation are bounded within the domain. 
