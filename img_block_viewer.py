#!/usr/bin/env python3

# Usage:
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/clip/RM006_s128_c13_f8906-9056.tif'
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/ims_based/z00060_c3_2.ims'

# TODO: vtkVRRenderWindow

import numpy as np
from numpy import sin, cos, pi

import tifffile
import h5py

import vtk

import pprint

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

    # TODO: determing this value automatically
    metadata['oblique_image'] = True if metadata['ImageLength']==788 else False

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

def read_ims(ims_path, level = 0):
    ims = h5py.File(ims_path, 'r')
    img = ims['DataSet']['ResolutionLevel %d'%(level)]['TimePoint 0']['Channel 0']['Data']
    print('image shape: ', img.shape, ' dtype =', img.dtype)

    # convert metadata in IMS to python dict
    u = ims['DataSetInfo']
    metadata = {}
    for it in ims['DataSetInfo'].keys():
        metadata[it] = \
            {k:''.join([c.decode('utf-8') for c in v])
                for k, v in u[it].attrs.items()}

    #img_clip = img[:, :, :]  # actually read the data
    #img_clip = np.transpose(np.array(img), (2,1,0))
    img_clip = np.array(img)

    metadata['imagej'] = {'voxel_size_um': '(1.0, 1.0, 1.0)'}
    metadata['oblique_image'] = False

    return img_clip, metadata

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
    r = np.array(cam1.GetPosition()) - np.array(cam1.GetFocalPoint())
    r = r / np.linalg.norm(r) * dist

    # Set also up direction?
    cam2.SetRoll(cam1.GetRoll())
    cam2.SetPosition(r)
    cam2.SetFocalPoint(0, 0, 0)
#    print(cam2.GetModelViewTransformMatrix())

def ModifiedCallbackFunction(caller, ev):
    # never happed
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

def Import3DImage(file_name, *item, **keys):
    # img_arr should be a numpy array with
    #   dimension order: Z C Y X  (full form TZCYXS)
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        img_arr, img_meta = read_tiff(file_name)
    elif file_name.endswith('.ims'):
        img_arr, img_meta = read_ims(file_name, *item, **keys)
    pprint.pprint(img_meta)
    return img_arr, img_meta

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
    img_arr, img_meta = Import3DImage(file_name, level=2)
    n_ch = 1

    voxel_size_um = img_meta['imagej']['voxel_size_um'][1:-1]
    voxel_size_um = tuple(map(float, voxel_size_um.split(', ')))

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
    b_oblique_correction = img_meta['oblique_image'] \
                               if 'oblique_image' in img_meta else False
    print('b_oblique_correction: ', b_oblique_correction)
    if b_oblique_correction:
        img_importer.SetDataSpacing(voxel_size_um[0], voxel_size_um[1],
                                    voxel_size_um[2]*np.sqrt(2))
        rotMat = [ \
            1.0, 0.0,            0.0,
            0.0, cos(45/180*pi), 0.0,
            0.0,-sin(45/180*pi), 1.0
        ]
        img_importer.SetDataDirection(rotMat)
    else:
        img_importer.SetDataSpacing(voxel_size_um)

#    print(img_importer.GetDataDirection())
#    print(img_importer.GetDataSpacing())

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
    #opacity_scale = 40.0
    opacity_scale = 10.0
    opacity_transfer_function = vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(opacity_scale*20, 0.0)
    opacity_transfer_function.AddPoint(opacity_scale*255, 0.2)

    # Create transfer mapping scalar value to color.
    #trans_scale = 40.0
    trans_scale = 10.0
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

    # Create the renderers
    renderer1 = vtkRenderer()
    renderer1.SetLayer(0)

    # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
    renderer2 = vtkRenderer()  # for axes
    renderer2.SetLayer(1)
    renderer2.SetViewport(0.0, 0.0, 0.2, 0.2)
    
    # vtkAssembly
    # https://vtk.org/doc/nightly/html/classvtkAssembly.html#details
    
    # Create the renderer window
    render_window = vtkRenderWindow()
#    render_window = vtkSynchronizedRenderWindows()
    render_window.AddRenderer(renderer1)
    render_window.AddRenderer(renderer2)

    # Create the interactor (for keyboard and mouse)
    interactor = vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(MyInteractorStyle())
    interactor.AddObserver('KeyPressEvent', KeypressCallbackFunction)
#    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
    interactor.AddObserver('InteractionEvent', ModifiedCallbackFunction)
    interactor.SetRenderWindow(render_window)

    # Create image object
    img_importer = ImportImage(file_name)

    volume = SetupVolumeRender(img_importer)

    # Create Axes object
    # vtkCubeAxesActor()
    # https://kitware.github.io/vtk-examples/site/Python/Visualization/CubeAxesActor/

    # Dynamically change position of Axes
    # https://discourse.vtk.org/t/dynamically-change-position-of-axes/691
    axes = vtkAxesActor()
    axes.SetTotalLength([1.0, 1.0, 1.0])
    axes.SetAxisLabels(False)

    # Append objects to renderers
    renderer1.AddVolume(volume)
    renderer1.SetBackground(colors.GetColor3d('Wheat'))
    renderer1.GetActiveCamera().Azimuth(45)
    renderer1.GetActiveCamera().Elevation(30)
    renderer1.ResetCameraClippingRange()
    renderer1.ResetCamera()

    renderer2.AddActor(axes)
    renderer2.GetActiveCamera().DeepCopy(renderer1.GetActiveCamera())
    renderer2.GetActiveCamera().SetClippingRange(0.1, 1000)
    AlignCameraDirection(renderer2.GetActiveCamera(),
                         renderer1.GetActiveCamera())

    # Set the render window
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
