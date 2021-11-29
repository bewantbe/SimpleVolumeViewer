#!/usr/bin/env python3

# Usage:
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/clip/RM006_s128_c13_f8906-9056.tif'

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter
)
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper
)
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper
from vtk import vtkTIFFReader, vtkImageReader
import vtk

import numpy as np
import tifffile
from numpy import sin, cos, pi

# copy from volumeio.py
# Read tiff file, return images and meta data
def read_tiff(tif_path, as_np_array = True):
    # see also https://pypi.org/project/tifffile/
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_val.name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    if as_np_array:
        images = tifffile.imread(tif_path)
    else:
        images = []
        for page in tif.pages:
            images.append(page.asarray())
    return images, metadata

# Read tiff file, return images and meta data
def read_tiff_meta(tif_path):
    # see also https://pypi.org/project/tifffile/
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    return metadata

# mouse interaction
class MyInteractorStyle(vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.AddObserver('MiddleButtonPressEvent', self.middle_button_press_event)
        self.AddObserver('MiddleButtonReleaseEvent', self.middle_button_release_event)

    def middle_button_press_event(self, obj, event):
        print('Middle Button pressed')
        self.OnMiddleButtonDown()
        return

    def middle_button_release_event(self, obj, event):
        print('Middle Button released')
        self.OnMiddleButtonUp()
        return

# import image to vtkImageImport() to have a connection
def ImportImage(fileName):
    # Ref:
    # Numpy 3D array into VTK data types for volume rendering?
    # https://discourse.vtk.org/t/numpy-3d-array-into-vtk-data-types-for-volume-rendering/3455/2
    # VTK Reading 8-bit tiff files (solved) VTK4.2
    # https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

    # code from Slicer
    # https://github.com/Slicer/Slicer/blob/2515768aaf70c161d781ff41f36f2a0655c88efb/Base/Python/slicer/util.py#L1950
    # def updateVolumeFromArray(volumeNode, img_arr):

    # See https://python.hotexamples.com/examples/vtk/-/vtkImageImport/python-vtkimageimport-function-examples.html
    img_arr, img_meta = read_tiff(fileName)
    print(img_meta)
    n_ch = 1

    imgImporter = vtk.vtkImageImport()
    simg = np.ascontiguousarray(img_arr, img_arr.dtype)  # maybe .flatten()
    # see also: SetImportVoidPointer
    imgImporter.CopyImportVoidPointer(simg.data, simg.nbytes)
    if img_arr.dtype == np.uint8:
        imgImporter.SetDataScalarTypeToUnsignedChar()
    elif img_arr.dtype == np.uint16:
        imgImporter.SetDataScalarTypeToUnsignedShort()
    else:
        raise "Unsupported format"
    imgImporter.SetNumberOfScalarComponents(n_ch)
    imgImporter.SetDataExtent (0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    imgImporter.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    #imgImporter.setDataOrigin()

    # the 3x3 matrix to rotate the coordinates from index space (ijk) to physical space (xyz)
    b_45d_correction = True
    if b_45d_correction:
        imgImporter.SetDataSpacing(1.0, 1.0, 3.5)
        rotMat = [ \
            1.0, 0.0,            0.0,
            0.0, cos(45/180*pi), 0.0,
            0.0,-sin(45/180*pi), 1.0
        ]
        imgImporter.SetDataDirection(rotMat)
    else:
        imgImporter.SetDataSpacing(1.0, 1.0, 2.5)

    print(imgImporter.GetDataDirection())
    print(imgImporter.GetDataSpacing())

    return imgImporter

def ShotScreen(renWin):
    # Take a screenshot
    # From: https://kitware.github.io/vtk-examples/site/Python/Utilities/Screenshot/
    w2if = vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName('TestScreenshot.png')
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

def SetupVolumeRender(imgImporter):
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
    #volumeMapper = vtkFixedPointVolumeRayCastMapper()
    volumeMapper = vtkGPUVolumeRayCastMapper()
    #volumeMapper.SetBlendModeToComposite()
    #volumeMapper.SetInputConnection(reader.GetOutputPort())
    # vtkVolumeMapper
    # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
    volumeMapper.SetInputConnection(imgImporter.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume

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
    iren.SetInteractorStyle(MyInteractorStyle())
    iren.SetRenderWindow(renWin)

    imgImporter = ImportImage(fileName)

    volume = SetupVolumeRender(imgImporter)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('Wheat'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    renWin.SetSize(2400, 1800)
    renWin.SetWindowName('SimpleRayCast')
    renWin.Render()

    ShotScreen(renWin)
    
    iren.Initialize()
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
