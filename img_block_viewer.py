#!/usr/bin/env python3

# Usage:
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/clip/RM006_s128_c13_f8906-9056.tif'

# TODO: vtkVRRenderWindow

import numpy as np
from numpy import sin, cos, pi

import tifffile

import vtk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleFlight,
    vtkInteractorStyleTerrain
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonTransforms import vtkTransform
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
# vtkInteractorStyleFlight
# vtkInteractorStyleTrackballCamera
class MyInteractorStyle(vtkInteractorStyleTerrain):

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

# Ref: Demonstrates the use of two renderers. Notice that the second (and subsequent) renderers will have a transparent background.
# https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
def KeypressCallbackFunction(caller, ev):
    iren = caller
    renderers = iren.GetRenderWindow().GetRenderers()
#    if renderers.GetNumberOfItems() < 2:
#        print('We need at least two renderers, we have only', renderers.GetNumberOfItems())
#        return
#    renderers.InitTraversal()
#    # Top item
#    ren0 = renderers.GetNextItem()
#    # Bottom item
#    ren1 = renderers.GetNextItem()

    key = iren.GetKeySym()

    if key == '0':
        print('Pressed:', key)
#        iren.GetRenderWindow().GetInteractor().GetInteractorStyle().SetDefaultRenderer(ren0)
#        ren0.InteractiveOn()
#        ren1.InteractiveOff()
    if key == '1':
        print('Pressed:', key)
#        iren.GetRenderWindow().GetInteractor().GetInteractorStyle().SetDefaultRenderer(ren1)
#        ren0.InteractiveOff()
#        ren1.InteractiveOn()

# align cam2 by cam1
# make cam2 dist away from origin
def AlignCameraDirection(cam2, cam1, dist=4.0):
    print(cam1)
    print(cam1.GetModelViewTransformMatrix())

    r = np.array(cam1.GetPosition()) - np.array(cam1.GetFocalPoint())
    r = r / np.linalg.norm(r) * dist

    # Set also up direction?
    cam2.SetRoll(cam1.GetRoll())
    cam2.SetPosition(r)
    cam2.SetFocalPoint(0, 0, 0)
    
    # cam2.SetUserViewTransform
    
    print(cam2)
    print(cam2.GetModelViewTransformMatrix())

#    cam2.SetRoll(cam1.GetRoll())
#    cam2.SetPosition(0.0, 0.0, 0.0)
    
#    cam2.SetUseExplicitProjectionTransformMatrix(True)
#    cam2.SetModelTransformMatrix(view_mat)
#    cam2.SetExplicitProjectionTransformMatrix(view_mat)
#    cam2.
#    cam2.SetRoll(cam1.GetRoll())
#    cam2.SetPosition(cam1.GetPosition())
#    cam2.SetViewUp(0, 1, 0)
#    cam2.ApplyTransform(view_mat)
#    ren2.SetFocalPoint(0.0, 0.0, 0.0)  # no
#    ren2.SetDistance(100.0)            # no
    
#    view_mat2 = cam2.GetModelViewTransformMatrix()
#    print(view_mat2)

def ModifiedCallbackFunction(caller, ev):
#    print(caller)
    print(ev)
    if ev == 'StartRotateEvent':
        pass
    elif ev == 'EndRotateEvent':
        pass
    elif ev == 'RotateEvent':
        pass
    
    rens = caller.GetRenderWindow().GetRenderers()
    rens.InitTraversal()
    ren1 = rens.GetNextItem()
    ren2 = rens.GetNextItem()
    cam1 = ren1.GetActiveCamera()
    cam2 = ren2.GetActiveCamera()

    AlignCameraDirection(cam2, cam1)
    
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

def ShotScreen(render_window):
    # Take a screenshot
    # From: https://kitware.github.io/vtk-examples/site/Python/Utilities/Screenshot/
    win2if = vtkWindowToImageFilter()
    win2if.SetInput(render_window)
    win2if.SetInputBufferTypeToRGB()
    win2if.ReadFrontBufferOff()
    win2if.Update()

    # If need transparency in a screenshot
    # https://stackoverflow.com/questions/34789933/vtk-setting-transparent-renderer-background
    
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
    renderer1 = vtkRenderer()
    renderer1.SetLayer(0)
    # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
    renderer2 = vtkRenderer()  # for axes
    renderer2.SetLayer(1)
    renderer2.SetViewport(0.0, 0.0, 0.2, 0.2)
    
    # vtkAssembly
    # https://vtk.org/doc/nightly/html/classvtkAssembly.html#details
    
    render_window = vtkRenderWindow()
#    render_window = vtkSynchronizedRenderWindows()
    render_window.AddRenderer(renderer1)
    render_window.AddRenderer(renderer2)

    interactor = vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(MyInteractorStyle())
    interactor.AddObserver('KeyPressEvent', KeypressCallbackFunction)
#    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
    interactor.AddObserver('InteractionEvent', ModifiedCallbackFunction)
    interactor.SetRenderWindow(render_window)

    img_importer = ImportImage(file_name)

    volume = SetupVolumeRender(img_importer)

    # vtkCubeAxesActor()
    # https://kitware.github.io/vtk-examples/site/Python/Visualization/CubeAxesActor/

    # Dynamically change position of Axes
    # https://discourse.vtk.org/t/dynamically-change-position-of-axes/691
#    transform = vtkTransform()
#    transform.Translate(100.0, 100.0, 100.0)
    axes = vtkAxesActor()
    axes.SetTotalLength([1.0, 1.0, 1.0])
#    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d('Red'))
#    axes.SetXAxisLabelText('test')
    axes.SetAxisLabels(False)

    renderer1.AddVolume(volume)
    renderer1.SetBackground(colors.GetColor3d('Wheat'))
    renderer1.GetActiveCamera().Azimuth(45)
    renderer1.GetActiveCamera().Elevation(30)
    renderer1.ResetCameraClippingRange()
    renderer1.ResetCamera()

    renderer2.AddActor(axes)
    print(renderer1.GetActiveCamera())
    #renderer2.SetActiveCamera(renderer1.GetActiveCamera())
    #renderer2.GetActiveCamera().ShallowCopy(renderer1.GetActiveCamera())
    renderer2.GetActiveCamera().DeepCopy(renderer1.GetActiveCamera())
    #renderer2.GetActiveCamera().SetFreezeFocalPoint(True)
    #renderer2.ResetCameraClippingRange()
    #renderer2.ResetCamera()
    renderer2.GetActiveCamera().SetClippingRange(0.1, 1000)
    
    AlignCameraDirection(renderer2.GetActiveCamera(),
                         renderer1.GetActiveCamera())


    render_window.SetSize(2400, 1800)
    render_window.SetWindowName('SimpleRayCast')
    render_window.SetNumberOfLayers(2)
    render_window.Render()

    ShotScreen(render_window)
    
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
