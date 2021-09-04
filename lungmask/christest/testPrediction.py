import sys
sys.path.append("../lungmask")

import SimpleITK as sitk
import mask
import matplotlib.pyplot as plt
from matplotlib import colors

reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName("../../AutLungSegm/datasetAqp/dataOkla/DataFinal/TRAIN/CT/coronacases_001.nii")
image_input = reader.Execute();

size = image_input.GetSize()
print("Image size:", image_input.GetSize())
print("Image direction:", image_input.GetDirection())
print("Image dimension:", image_input.GetDimension())
print("Image origin:", image_input.GetOrigin())

#############################################################
print("Aplicando prediccion con el modelo...")
output_mask = mask.apply(image_input,batch_size=7)
print("Finalizando prediccion\n")
#############################################################


fig = plt.figure(figsize=(12,12))
plt.imshow(output_mask[40],norm=colors.Normalize(0,1))
plt.savefig("mask_out.png")


newMask = sitk.GetImageFromArray(output_mask)
newMask.CopyInformation(image_input)


print("Image size:", newMask.GetSize())
print("Image direction:", newMask.GetDirection())
print("Image dimension:", newMask.GetDimension())
print("Image origin:", newMask.GetOrigin())

writer = sitk.ImageFileWriter()
writer.SetFileName("mascaraFinal.nii")
writer.Execute(newMask);








