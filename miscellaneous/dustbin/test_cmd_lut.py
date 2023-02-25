from vtkmodules.util.numpy_support import numpy_to_vtk

s = swcs[0]

mapper    = s.actor.GetMapper()     # vtkPolyDataMapper
poly_data = mapper.GetInput()          # vtkPolyData
seg_data  = poly_data.GetCellData()    # vtkCellData
n_seg = poly_data.GetNumberOfCells()
# give random scalars (color)
n_color = 10
scalar_color = (np.random.randint(0,n_color, (n_seg,)) + 0.5) / n_color
seg_data.SetScalars(numpy_to_vtk(scalar_color, deep=True))
# set colormap (lookup table)
lut = vtkLookupTable()
lut.SetNumberOfTableValues(402)   # 402 (256*pi/2) as suggested by lut.SetRampToSCurve()
lut.Build()
mapper.SetLookupTable(lut)
mapper.SetColorModeToMapScalars()  # use lookup table
#mapper.SetColorModeToMapScalars()  # the scalar data is color
mapper.Modified()