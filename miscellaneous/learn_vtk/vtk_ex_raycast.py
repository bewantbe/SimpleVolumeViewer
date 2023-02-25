#!/usr/bin/env python

# help from https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

# python vtk_ex_raycast.py '/media/xyy/DATA/VISoRData/RM06_set2020-09-19/RM06_s128_c13_f8906_p3.tif'

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper
from vtk import vtkTIFFReader, vtkImageReader
import vtk

def main():
    fileName = get_program_parameters()

    colors = vtkNamedColors()

    # This is a simple volume rendering example that
    # uses a vtkFixedPointVolumeRayCastMapper

    # Create the standard renderer, render window
    # and interactor.
    ren1 = vtkRenderer()

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren1)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data.
#    reader = vtkStructuredPointsReader()
#    reader.SetFileName(fileName)

    reader = vtkTIFFReader()
    reader.SetFileName(fileName)
    reader.Update()
    
    reader.GetDimensions()

    # Create transfer mapping scalar value to opacity.
    opacity_scale = 40.0
    opacityTransferFunction = vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(opacity_scale*20, 0.0)
    opacityTransferFunction.AddPoint(opacity_scale*255, 0.2)

    # Create transfer mapping scalar value to color.
    trans_scale = 40.0
    colorTransferFunction = vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(trans_scale*0.0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*64.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*128.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(trans_scale*192.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*255.0, 0.0, 0.2, 0.0)

    # The property describes how the data will look.
    volumeProperty = vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    #volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.SetInterpolationType(vtk.VTK_CUBIC_INTERPOLATION)

    # The mapper / ray cast function know how to render the data.
    volumeMapper = vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('Wheat'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    renWin.SetSize(2400, 1800)
    renWin.SetWindowName('SimpleRayCast')
    renWin.Render()

    iren.Start()


def get_program_parameters():
    import argparse
    description = 'Volume rendering of a high potential iron protein.'
    epilogue = '''
    This is a simple volume rendering example that uses a vtkFixedPointVolumeRayCastMapper.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filename', help='ironProt.vtk', default='/media/xyy/DATA/VISoRData/RM06_set2020-09-19/RM06_s128_c13_f8906_p3.tif')
    args = parser.parse_args()
    return args.filename


if __name__ == '__main__':
    main()
