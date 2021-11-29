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
def ImportImage(file_name):
    # Ref:
    # Numpy 3D array into VTK data types for volume rendering?
    # https://discourse.vtk.org/t/numpy-3d-array-into-vtk-data-types-for-volume-rendering/3455/2
    # VTK Reading 8-bit tiff files (solved) VTK4.2
    # https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

    # code from Slicer
    # https://github.com/Slicer/Slicer/blob/2515768aaf70c161d781ff41f36f2a0655c88efb/Base/Python/slicer/util.py#L1950
    # def updateVolumeFromArray(volumeNode, img_arr):

    # See https://python.hotexamples.com/examples/vtk/-/vtkImageImport/python-vtkimageimport-function-examples.html
    img_arr, img_meta = read_tiff(file_name)
    print(img_meta)
    n_ch = 1

    img_importer = vtk.vtkImageImport()
    simg = np.ascontiguousarray(img_arr, img_arr.dtype)  # maybe .flatten()
    # see also: SetImportVoidPointer
    img_importer.CopyImportVoidPointer(simg.data, simg.nbytes)
    if img_arr.dtype == np.uint8:
        img_importer.SetDataScalarTypeToUnsignedChar()
    elif img_arr.dtype == np.uint16:
        img_importer.SetDataScalarTypeToUnsignedShort()
    else:
        raise "Unsupported format"
    img_importer.SetNumberOfScalarComponents(n_ch)
    img_importer.SetDataExtent (0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    img_importer.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    #img_importer.setDataOrigin()

    # the 3x3 matrix to rotate the coordinates from index space (ijk) to physical space (xyz)
    b_45d_correction = True
    if b_45d_correction:
        img_importer.SetDataSpacing(1.0, 1.0, 3.5)
        rotMat = [ \
            1.0, 0.0,            0.0,
            0.0, cos(45/180*pi), 0.0,
            0.0,-sin(45/180*pi), 1.0
        ]
        img_importer.SetDataDirection(rotMat)
    else:
        img_importer.SetDataSpacing(1.0, 1.0, 2.5)

    print(img_importer.GetDataDirection())
    print(img_importer.GetDataSpacing())

    return img_importer

def ShotScreen(ren_win):
    # Take a screenshot
    # From: https://kitware.github.io/vtk-examples/site/Python/Utilities/Screenshot/
    win2if = vtkWindowToImageFilter()
    win2if.SetInput(ren_win)
    win2if.SetInputBufferTypeToRGB()
    win2if.ReadFrontBufferOff()
    win2if.Update()

    writer = vtkPNGWriter()
    writer.SetFileName('TestScreenshot.png')
    writer.SetInputConnection(win2if.GetOutputPort())
    writer.Write()

def SetupVolumeRender(img_importer):
    # Create transfer mapping scalar value to opacity.
    opacity_scale = 40.0
    opacity_transfer_function = vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(opacity_scale*20, 0.0)
    opacity_transfer_function.AddPoint(opacity_scale*255, 0.2)

    # Create transfer mapping scalar value to color.
    trans_scale = 40.0
    color_transfer_function = vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(trans_scale*0.0, 0.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(trans_scale*64.0, 1.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(trans_scale*128.0, 0.0, 0.0, 1.0)
    color_transfer_function.AddRGBPoint(trans_scale*192.0, 0.0, 1.0, 0.0)
    color_transfer_function.AddRGBPoint(trans_scale*255.0, 0.0, 0.2, 0.0)

    # The property describes how the data will look.
    volume_property = vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.ShadeOn()
    #volume_property.SetInterpolationTypeToLinear()
    volume_property.SetInterpolationType(vtk.VTK_CUBIC_INTERPOLATION)

    # The mapper / ray cast function know how to render the data.
    #volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper = vtkGPUVolumeRayCastMapper()
    #volume_mapper.SetBlendModeToComposite()
    #volume_mapper.SetInputConnection(reader.GetOutputPort())
    # vtkVolumeMapper
    # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
    volume_mapper.SetInputConnection(img_importer.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

def main():
    file_name = get_program_parameters()

    colors = vtkNamedColors()

    # This is a simple volume rendering example that
    # uses a vtkFixedPointVolumeRayCastMapper

    # Create the standard renderer, render window
    # and interactor.
    ren1 = vtkRenderer()

    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren1)

    interactor = vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(MyInteractorStyle())
    interactor.SetRenderWindow(ren_win)

    img_importer = ImportImage(file_name)

    volume = SetupVolumeRender(img_importer)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('Wheat'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    ren_win.SetSize(2400, 1800)
    ren_win.SetWindowName('SimpleRayCast')
    ren_win.Render()

    ShotScreen(ren_win)
    
    interactor.Initialize()
    interactor.Start()


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
