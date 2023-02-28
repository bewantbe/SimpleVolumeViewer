# Construct and test vtk ColorMap

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from vtkmodules.vtkCommonCore import vtkLookupTable

# vtkLookupTable
# https://vtk.org/doc/nightly/html/classvtkLookupTable.html
# VTK/Common/Core/vtkLookupTable.cxx
# https://github.com/Kitware/VTK/blob/d706250a1422ae1e7ece0fa09a510186769a5fec/Common/Core/vtkLookupTable.cxx

lut = vtkLookupTable()
print(lut.GetNumberOfTableValues())
lut.SetNumberOfTableValues(402)   # 402 (256*pi/2) as suggested by SetRampToSCurve()
#lut.SetNumberOfTableValues(402)   # 402 (256*pi/2) as suggested by SetRampToSCurve()
#lut.SetHueRange(0.0, 0.66667)      # H
#lut.SetSaturationRange(1.0, 1.0)   # S
#lut.SetValueRange(1.0, 1.0)        # V
#lut.SetAlphaRange(1.0, 1.0)
#lut.SetNanColor(0.5, 0.0, 0.0, 1.0)
#lut.SetBelowRangeColor(0.0, 0.0, 0.0, 1.0)
#lut.SetAboveRangeColor(1.0, 1.0, 1.0, 1.0)
#lut.SetTableRange(0.0, 1.0)
##lut.SetRampToLinear()
#lut.SetRampToSCurve()
#lut.SetScaleToLinear()
lut.Build()
# or set the table ourself
#lut.SetTableValue(i, r, g, b, a)

n_c = 32
v = np.zeros((n_c, 3))
vals = np.zeros(n_c)
a = [0,0,0]
for i in range(n_c):
    val = float(i)/n_c
    lut.GetColor(val, a)
    v[i, :] = a
    vals[i] = lut.GetIndex(val)
    print(a)

n_h = 10
v_img = repmat(v.reshape(n_c*3), n_h, 1).reshape(n_h, n_c, 3)

plt.figure(1)
plt.plot(vals)

plt.figure(2)
plt.plot(v)

plt.figure(3)
plt.imshow(v_img)

plt.show()
