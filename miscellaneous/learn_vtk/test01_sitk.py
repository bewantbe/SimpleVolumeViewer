
import SimpleITK as sitk
import numpy as np

a = np.zeros((2,3))

scale = sitk.ScaleTransform(2, (1/3.5, 1))

img0 = sitk.ReadImage('/home/xyy/code/py/vtk_test/TestScreenshot.png')

#img = sitk.GetImageFromArray(a)
img2 = sitk.Resample(img0, scale)

print(img2.GetSize())
# 3 2

sitk.WriteImage(img2, '/home/xyy/code/py/vtk_test/TestScreenshot_sitk2.png')


