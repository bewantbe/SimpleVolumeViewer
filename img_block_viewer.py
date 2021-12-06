#!/usr/bin/env python3

# Usage:
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/clip/RM006_s128_c13_f8906-9056.tif'
# python img_block_viewer.py --filename '/media/xyy/DATA/RM006_related/ims_based/z00060_c3_2.ims'

# TODO: vtkVRRenderWindow

import os
import json
import pprint

import numpy as np
from numpy import sin, cos, pi

import tifffile
import h5py

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

def DefaultGUIConfigure():
    d = {
        "window": {
            "size": [2400, 1800],
            "title": "SimpleRayCast",
            "number_of_layers": 2
        },

        "renderers":{
            "0":{
                "layer": 0
            },
            "1":{
                "layer": 1,
                "view_port": [0.0, 0.0, 0.2, 0.2]
            }
        }
    }
    return d

def DefaultScene():
    d = {
        "object_properties": {
            "volume": {
                "opacity_transfer_function": {
                    "AddPoint": [
                        [20, 0.0],
                        [255, 0.2]
                    ],
                    "opacity_scale": 40.0
                },
                "color_transfer_function": {
                    "AddRGBPoint": [
                        [0.0, 0.0, 0.0, 0.0],
                        [64.0, 1.0, 0.0, 0.0],
                        [128.0, 0.0, 0.0, 1.0],
                        [192.0, 0.0, 1.0, 0.0],
                        [255.0, 0.0, 0.2, 0.0]
                    ],
                    "trans_scale": 40.0
                },
                "interpolation": "cubic"
            }
        },

        "objects": {
            "background": {
                "type": "Background",
                "color": "Wheat"
            },
            "axes": {
                "type": "AxesActor",
                "ShowAxisLabels": "False",
                "renderer": "1"
            },
            "camera1": {
                "type": "Camera",
                "renderer": "0",
                "Azimuth": 45,
                "Elevation": 30,
                "clipping_range": [0.1, 5000]
            },
            "camera2": {
                "type": "Camera",
                "renderer": "1",
                "follow_direction": "camera1"
            }
        }
    }
    return d

debug_level = 4

# Used for print error, controlled by debug_level.
# higher debug_level will show more info.
# 0 == debug_level will show no info.
def dbg_print(level, *p, **keys):
    if level > debug_level:
        return
    level_str = {1:"Error", 2:"Warning", 3:"Hint", 4:"Message"}
    print(level_str[level] + ":", *p, **keys)

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

def read_ims(ims_path, extra_conf):
    ims = h5py.File(ims_path, 'r')
    level      = int(extra_conf.get('level', 0))
    channel    = int(extra_conf.get('channel', 0))
    time_point = int(extra_conf.get('time_point', 0))
    img = ims['DataSet']['ResolutionLevel %d'%(level)] \
                        ['TimePoint %d'%(time_point)] \
                        ['Channel %d'%(channel)]['Data']
    dbg_print(4, 'image shape: ', img.shape, ' dtype =', img.dtype)

    # convert metadata in IMS to python dict
    img_info = ims['DataSetInfo']
    metadata = {'read_ims':
        {'level': level, 'channel': channel, 'time_point': time_point}}
    for it in img_info.keys():
        metadata[it] = \
            {k:''.join([c.decode('utf-8') for c in v])
                for k, v in img_info[it].attrs.items()}

    #img_clip = np.transpose(np.array(img), (2,1,0))
    img_clip = np.array(img)         # actually read the data

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

def Read3DImageDataFromFile(file_name, *item, **keys):
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        img_arr, img_meta = read_tiff(file_name)
    elif file_name.endswith('.ims'):
        img_arr, img_meta = read_ims(file_name, *item, **keys)
    pprint.pprint(img_meta)
    return img_arr, img_meta

# import image to vtkImageImport() to have a connection
# img_arr must be a numpy-like array
#   dimension order: Z C Y X  (full form TZCYXS)
# img_meta may contain
#   img_meta['imagej']['voxel_size_um']
#   img_meta['oblique_image']
def ImportImageArray(img_arr, img_meta):
    # Ref:
    # Numpy 3D array into VTK data types for volume rendering?
    # https://discourse.vtk.org/t/numpy-3d-array-into-vtk-data-types-for-volume-rendering/3455/2
    # VTK Reading 8-bit tiff files (solved) VTK4.2
    # https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

    # code from Slicer
    # https://github.com/Slicer/Slicer/blob/2515768aaf70c161d781ff41f36f2a0655c88efb/Base/Python/slicer/util.py#L1950
    # def updateVolumeFromArray(volumeNode, img_arr):

    # See https://python.hotexamples.com/examples/vtk/-/vtkImageImport/python-vtkimageimport-function-examples.html

    # Wild guess number of channels
    if len(img_arr.shape) == 4:
        n_ch = img_arr.shape[1]
    else:
        n_ch = 1

    if ('imagej' in img_meta) and \
       ('voxel_size_um' in img_meta['imagej']):
        if isinstance(img_meta['imagej']['voxel_size_um'], str):
            voxel_size_um = img_meta['imagej']['voxel_size_um'][1:-1]
            voxel_size_um = tuple(map(float, voxel_size_um.split(', ')))
        else:  # assume array
            voxel_size_um = img_meta['imagej']['voxel_size_um']

    img_importer = vtk.vtkImageImport()
    simg = np.ascontiguousarray(img_arr, img_arr.dtype)  # maybe .flatten()?
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
    b_oblique_correction = img_meta.get('oblique_image', False)
    dbg_print(4, 'b_oblique_correction: ', b_oblique_correction)
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

# import image to vtkImageImport() to have a connection
# extra_conf for extra setting to extract the image
# the extra_conf takes higher priority than meta data in the file
def ImportImageFile(file_name, extra_conf = None):
    img_arr, img_meta = Read3DImageDataFromFile(file_name, extra_conf)
    img_import = ImportImageArray(img_arr, img_meta)
    return img_import

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

def MergeFullDict(d_contain, d_update):
    # update dict d_contain by d_update
    # i.e. overwrite d_contain for items exist in d_update
    # Ref. https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-take-union-of-dictionari
    def DeepUpdate(d_contain, d_update):
        for key, value in d_update.items(): 
            if key not in d_contain:
                d_contain[key] = value
            else:  # key in d_contain
                if isinstance(value, dict):
                    DeepUpdate(d_contain[key], value)
                else:  # overwirte
                    # simple sanity check: data type must agree
                    if type(d_contain[key]) == type(value):
                        d_contain[key] = value
                    else:
                        dbg_print(2, "DeepUpdate()", "key type mismatch! value discard.")
        return d_contain

    DeepUpdate(d_contain, d_update)

    return d_contain

# return a name not occur in name_set
def GetNonconflitName(prefix, name_set):
    i = 1
    while prefix in name_set:
        prefix = prefix + ".%.3d"%i
        i += 1
    return prefix

def ReadGUIConfigure(self, gui_conf_path):
    conf = DefaultGUIConfigure()
    if os.path.isfile(gui_conf_path):
        conf_ext = json.loads(open(gui_conf_path).read())
        MergeFullDict(conf, conf_ext)
    return conf

def ReadScene(self, scene_file_path):
    scene = DefaultScene()
    if os.path.isfile(scene_file_path):
        scene_ext = json.loads(open(scene_file_path).read())
        MergeFullDict(scene, scene_ext)
    return scene

class GUIControl:
    def __init__(self):
        # Load configure
        file_name = get_program_parameters()

        self.renderers = {}
        self.render_window = None
        self.interactor = None
        self.object_properties = {}
        self.scene_objects = {}
        
        # load default settings
        self.GUISetup(DefaultGUIConfigure())
        self.AppendToScene(DefaultScene())

    # setup window, renderers and interactor
    def GUISetup(self, gui_conf):
        dbg_print(4, gui_conf)
        if "window" in gui_conf:
            # TODO: stop the old window?
            if self.render_window is None:
                self.render_window = vtkRenderWindow()
            win_conf = gui_conf["window"]
            if "size" in win_conf:
                self.render_window.SetSize(win_conf["size"])
            if "title" in win_conf:
                self.render_window.SetWindowName(win_conf["title"])
            if "number_of_layers" in win_conf:
                self.render_window.SetNumberOfLayers(
                    win_conf["number_of_layers"])

        if "renderers" in gui_conf:
            # get our renderer list
            renderers = self.renderers
            # load new renderers
            for key, ren_conf in gui_conf["renderers"].items():
                if key in renderers:
                    # remove old renderer
                    self.render_window.RemoveRenderer(renderers[key])
                # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
                # setup new renderer
                renderer = vtkRenderer()
                if "layer" in ren_conf:
                    renderer.SetLayer(ren_conf["layer"])
                if "view_port" in ren_conf:
                    renderer.SetViewport(ren_conf["view_port"])
                renderers[key] = renderer
                # add new renderer to window
                self.render_window.AddRenderer(renderer)

        # Create the interactor (for keyboard and mouse)
        interactor = vtkRenderWindowInteractor()
        interactor.SetInteractorStyle(MyInteractorStyle())
        interactor.AddObserver('KeyPressEvent', KeypressCallbackFunction)
    #    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
        interactor.AddObserver('InteractionEvent', ModifiedCallbackFunction)
        interactor.SetRenderWindow(self.render_window)
        self.interactor = interactor

    # The property describes how the data will look.
    def AddObjectProperty(self, name, prop_conf):
        if name in self.object_properties:
            # TODO: do we need to remove old mappers?
            pass
        dbg_print(4, name, ':', prop_conf)
        if name.startswith("volume"):
            volume_property = vtkVolumeProperty()

            if 'opacity_transfer_function' in prop_conf:
                otf_conf = prop_conf['opacity_transfer_function']
                otf_v = otf_conf['AddPoint']
                otf_s = otf_conf['opacity_scale']
                # Create transfer mapping scalar value to opacity.
                otf = vtkPiecewiseFunction()
                otf.AddPoint(otf_s*otf_v[0][0], otf_v[0][1])
                otf.AddPoint(otf_s*otf_v[1][0], otf_v[1][1])
                volume_property.SetScalarOpacity(otf)

            if 'color_transfer_function' in prop_conf:
                ctf_conf = prop_conf['color_transfer_function']
                ctf_v = ctf_conf['AddRGBPoint']
                ctf_s = ctf_conf['trans_scale']
                for v in ctf_v:
                    v[0] = v[0] *  ctf_s
                # Create transfer mapping scalar value to color.
                ctf = vtkColorTransferFunction()
                for v in ctf_v:
                    ctf.AddRGBPoint(*v)
                volume_property.SetColor(ctf)

            volume_property.ShadeOn()

            if 'interpolation' in prop_conf:
                if prop_conf['interpolation'] == "cubic":
                    volume_property.SetInterpolationType(
                        vtk.VTK_CUBIC_INTERPOLATION)
                elif prop_conf['interpolation'] == "cubic":
                    volume_property.SetInterpolationTypeToLinear()
                else:
                    dbg_print(2, "AddObjectProperty(): unknown interpolation type")
            object_property = volume_property
        else:
            dbg_print(2, "AddObjectProperty(): unknown object type")

        self.object_properties.update({name: object_property})

    def AddObjects(self, name, obj_conf):
        if name in self.scene_objects:
            # TODO: do we need to remove old object?
            name = GetNonconflitName(name, self.scene_objects)

        renderer = self.renderers[
            obj_conf.get('renderer', '0')]

        if obj_conf['type'] == 'volume':
            dbg_print(2, "AddObjects: ",  obj_conf)
            dbg_print(2, "renderer: ",  obj_conf.get('renderer', '0'))
            # vtkVolumeMapper
            # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
            if obj_conf['mapper'] == 'GPUVolumeRayCastMapper':
                volume_mapper = vtkGPUVolumeRayCastMapper()
            elif obj_conf['mapper'] == 'FixedPointVolumeRayCastMapper':
                volume_mapper = vtkFixedPointVolumeRayCastMapper()
            else:
                volume_mapper = vtkGPUVolumeRayCastMapper()
            #volume_mapper.SetBlendModeToComposite()

            file_path = obj_conf['file_path']
            img_importer = ImportImageFile(file_path, obj_conf)
            volume_mapper.SetInputConnection(img_importer.GetOutputPort())

            volume_property = self.object_properties[
                obj_conf.get('property', 'volume')]

            # The volume holds the mapper and the property and
            # can be used to position/orient the volume.
            volume = vtkVolume()
            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            
            renderer.AddVolume(volume)

            if ('view_point' in obj_conf) and \
                obj_conf['view_point'] == 'auto':
                # auto view all actors
                renderer.ResetCamera()

            scene_object = volume

        elif obj_conf['type'] == 'AxesActor':
            # Create Axes object
            # vtkCubeAxesActor()
            # https://kitware.github.io/vtk-examples/site/Python/Visualization/CubeAxesActor/

            # Dynamically change position of Axes
            # https://discourse.vtk.org/t/dynamically-change-position-of-axes/691
            axes = vtkAxesActor()
            axes.SetTotalLength([1.0, 1.0, 1.0])
            axes.SetAxisLabels("true"==obj_conf.get('ShowAxisLabels', "False").lower())

            renderer.AddActor(axes)

            scene_object = axes

        elif obj_conf['type'] == 'Background':
            colors = vtkNamedColors()
            renderer.SetBackground(colors.GetColor3d(obj_conf['color']))
            scene_object = renderer

        elif obj_conf['type'] == 'Camera':
            if 'renderer' in obj_conf:
                cam = renderer.GetActiveCamera()
                renderer.ResetCameraClippingRange()
                renderer.ResetCamera()
            else:
                cam = vtk.vtkCamera()

            if ('Azimuth' in obj_conf) or ('Elevation' in obj_conf):
                cam.Azimuth(obj_conf['Azimuth'])
                cam.Elevation(obj_conf['Elevation'])

            if 'clipping_range' in obj_conf:
                dbg_print(4, 'AddObjects(): clipping_range', obj_conf['clipping_range'])
                cam.SetClippingRange(obj_conf['clipping_range'])

            if 'follow_direction' in obj_conf:
                cam_ref = self.scene_objects[obj_conf['follow_direction']]
                cam.DeepCopy(cam_ref)
                cam.SetClippingRange(0.1, 1000)
                AlignCameraDirection(cam, cam_ref)
            scene_object = cam

        self.scene_objects.update({name: scene_object})

    # add objects to the renderers
    def AppendToScene(self, scene_conf):
        if "object_properties" in scene_conf:
            for key, prop_conf in scene_conf["object_properties"].items():
                self.AddObjectProperty(key, prop_conf)

        if "objects" in scene_conf:
            for key, obj_conf in scene_conf["objects"].items():
                self.AddObjects(key, obj_conf)
        # see also vtkAssembly
        # https://vtk.org/doc/nightly/html/classvtkAssembly.html#details
        return

    def EasyObjectImporter(self, obj_desc):
        if not obj_desc:
            return
        if isinstance(obj_desc, str):
            obj_desc = {'filename': obj_desc}
        file_path = obj_desc['filename']
        if file_path.endswith('.tif'):
            # assume this a volume
            obj_conf = {
                "type": "volume",
                "mapper": "GPUVolumeRayCastMapper",
                "view_point": "auto",
                "file_path": file_path
            }
            name = GetNonconflitName('volume', self.scene_objects.keys())
            self.AddObjects(name, obj_conf)
        elif file_path.endswith('.ims') or file_path.endswith('.h5'):
            # assume this a volume
            obj_conf = {
                "type": "volume",
                "mapper": "GPUVolumeRayCastMapper",
                "view_point": "auto",
                "file_path": file_path,
                "level": obj_desc.get('level', 0)
            }
            name = GetNonconflitName('volume', self.scene_objects.keys())
            self.AddObjects(name, obj_conf)
        else:
            dbg_print(1, "Unreconized source format.")
        return obj_conf

    def ShotScreen(self):
        ShotScreen(self.render_window)

    def Start(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

def get_program_parameters():
    import argparse
    description = 'Simple volume viewer.'
    epilogue = '''
    This is a simple volume rendering viewer.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filename', help='image stack filepath')
    parser.add_argument('--level', help='for multi-level image (.ims), load only that level')
    args = parser.parse_args()
    # convert class attribute to dict
    keys = ['filename', 'level']
    d = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    return d

if __name__ == '__main__':
    gui = GUIControl()
    gui.EasyObjectImporter(get_program_parameters())
    gui.Start()
    
