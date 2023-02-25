    #  Get an outline of the data set for context.
    outline = vtkOutlineFilter()
    outline.SetInputConnection(reader.GetOutputPort())
    outlineMapper = vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetColor(colors.GetColor3d('Black'))


