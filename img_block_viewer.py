#!/usr/bin/env python3

# Usage:
# python img_block_viewer.py --filepath '/media/xyy/DATA/RM006_related/clip/RM006_s128_c13_f8906-9056.tif'
# ./img_block_viewer.py --filepath '/media/xyy/DATA/RM006_related/ims_based/z00060_c3_2.ims' --level 3 --range '[400:800, 200:600, 300:700]' --colorscale 10
# ./img_block_viewer.py --filepath /media/xyy/DATA/RM006_related/test_fanxiaowei_2021-12-14/3864-3596-2992_C3.ims --colorscale 10 --swc /media/xyy/DATA/RM006_related/test_fanxiaowei_2021-12-14/R2-N1-A2.json.swc_modified.swc --fibercolor green

# Python Wrappers for VTK
# https://vtk.org/doc/nightly/html/md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html

# Demonstrates physically based rendering using image based lighting and a skybox.
# https://kitware.github.io/vtk-examples/site/Python/Rendering/PBR_Skybox/

import os
import time
import json
import pprint

import numpy as np
from numpy import sin, cos, pi

import tifffile
import h5py

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonCore import (
    vtkPoints,
    VTK_CUBIC_INTERPOLATION
)
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import (
    vtkPiecewiseFunction,
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)
from vtkmodules.vtkIOImage import (
    vtkPNGWriter,
    vtkImageImport
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleFlight,
    vtkInteractorStyleTerrain,
    vtkInteractorStyleUser
)
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
    vtkActor,
    vtkPolyDataMapper
)
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper
)
# 
# noinspection PyUnresolvedReferences
#import vtkRenderingOpenGL2, vtkRenderingFreeType, vtkInteractionStyle
#import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from vtk.util.numpy_support import numpy_to_vtk

def DefaultGUIConfig():
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

def DefaultSceneConfig():
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
            },
            "axes": {
                "type": "AxesActor",
                "ShowAxisLabels": "False",
                "renderer": "1"
            },
#            "orientation": {
#                "type": "OrientationMarker",
#                "ShowAxisLabels": "False",
#                "renderer": "0"
#            },
        }
    }
    return d

debug_level = 2

# Used for print error, controlled by debug_level.
# higher debug_level will show more info.
# 0 == debug_level will show no info.
def dbg_print(level, *p, **keys):
    if level > debug_level:
        return
    level_str = {1:"Error", 2:"Warning", 3:"Hint", 4:"Message"}
    print(level_str[level] + ":", *p, **keys)

# Utilizer to convert a fraction to integer range
# mostly copy from VISoR_select_light/pick_big_block/volumeio.py
# Examples:
#   rg=[(1, 2)], max_pixel=100: return ( 0,  50)
#   rg=[(2, 2)], max_pixel=100: return (50, 100)
#   rg=[],       max_pixel=100: return (0, 100)
#   rg=(0, 50),  max_pixel=100: return ( 0,  50)
#   rg=([0.1], [0.2]), max_pixel=100: return ( 10,  20)
def rg_part_to_pixel(rg, max_pixel):
    if len(rg) == 0:
        return (0, max_pixel)
    elif len(rg)==1 and len(rg[0])==2:
        # in the form rg=[(1, 2)], it means 1/2 part of a range
        rg = rg[0]
        erg = (int((rg[0]-1)/rg[1] * max_pixel), 
               int((rg[0]  )/rg[1] * max_pixel))
        return erg
    elif len(rg)==2 and isinstance(rg[0], (list, tuple)):
        # in the form rg=([0.1], [0.2]), means 0.1~0.2 part of a range
        p0, p1 = rg[0][0], rg[1][0]
        erg = [int(p0 * max_pixel), int(p1 * max_pixel)]
        return erg
    else:  # return as-is
        return rg

def slice_from_str(slice_str):
    # Construct array slice object.
    # Ref: https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
    # Format example: [100:400, :, 20:]
    dim_ranges = slice_str[1:-1].split(',')
    # convert a:b:c to slice(a,b,c)
    dim_ranges = tuple(
                     slice(
                         *map(
                             lambda x: int(x.strip())
                                 if x.strip() else None,
                             rg.split(':')
                         ))
                     for rg in dim_ranges
                 )
    return dim_ranges

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
# Returm image array and metadata.
def read_tiff_meta(tif_path):
    # see also https://pypi.org/project/tifffile/
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    return metadata

# Read Imaris compatible image file.
# Returm image array and metadata.
def read_ims(ims_path, extra_conf = {}, cache_reader_obj = False):
    # TODO: how to impliment cache_reader_obj?
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

    dbg_print(3, 'read_ims(): extra_conf =', extra_conf)
    dim_ranges = slice_from_str(str(extra_conf.get('range', '[:,:,:]')))
    dbg_print(3, 'dim_ranges', dim_ranges)
    
    t0 = time.time()
    img_clip = np.array(img[dim_ranges])         # actually read the data
    dbg_print(3, "read_ims(): img read time: %6.3f" % (time.time()-t0))
    #img_clip = np.transpose(np.array(img_clip), (2,1,0))

    metadata['imagej'] = {'voxel_size_um': '(1.0, 1.0, 1.0)'}
    metadata['oblique_image'] = False

    return img_clip, metadata

# mouse interaction
# vtkInteractorStyleTerrain
# vtkInteractorStyleFlight
# vtkInteractorStyleTrackballCamera
# vtkInteractorStyleUser
class MyInteractorStyle(vtkInteractorStyleTerrain):

    def __init__(self, iren, guictrl):
        self.AddObserver('MiddleButtonPressEvent', self.middle_button_press_event)
        self.AddObserver('MiddleButtonReleaseEvent', self.middle_button_release_event)
        self.AddObserver('CharEvent', self.OnChar)
        self.iren = iren
        self.guictrl = guictrl

    def middle_button_press_event(self, obj, event):
        print('Middle Button pressed')
        self.OnMiddleButtonDown()
        return

    def middle_button_release_event(self, obj, event):
        print('Middle Button released')
        self.OnMiddleButtonUp()
        return

    def OnChar(self, obj, event):
        iren = self.iren

        key_sym  = iren.GetKeySym()   # useful for PageUp etc.
        key_code = iren.GetKeyCode()
        b_C = iren.GetControlKey()
        b_A = iren.GetAltKey()
        b_S = iren.GetShiftKey()  # sometimes reflected in key_code

        key_combo = ("Ctrl+" if b_C else "") + ("Alt+" if b_A else "") + ("Shift+" if b_S else "") + key_code
        dbg_print(4, 'Pressed:', key_combo, '  key_sym:', key_sym)
        
        is_default_binding = (key_code.lower() in 'jtca3efprsuw') and \
                             not b_C

        #        shift ctrl alt
        #    q    T    F    T
        #    3    F    F    T
        #    e    T    F    T
        #    r    T    F    T

        rens = iren.GetRenderWindow().GetRenderers()
        
        if key_combo == 'r':
            rens.InitTraversal()
            ren1 = rens.GetNextItem()
            ren2 = rens.GetNextItem()
            cam1 = ren1.GetActiveCamera()
            cam2 = ren2.GetActiveCamera()
            rotator = execSmoothRotation(cam1, 60.0)
            timerHandler(iren, 6.0, rotator).start()
        elif key_combo == '+' or key_combo == '-':
            vol_name = 'volume'
            obj_prop = self.guictrl.object_properties[vol_name]
            cs_o, cs_c = GetColorScale(obj_prop)
            k = np.sqrt(np.sqrt(2))
            if key_combo == '+':
                k = 1.0 / k
            SetColorScale(obj_prop, [cs_o*k, cs_c*k])
            scene_obj = self.guictrl.scene_objects['volume']
#            scene_obj.Modified()  # not work
#            scene_obj.Update()
            iren.GetRenderWindow().Render()

        # Let's say, disable all default key bindings (except q)
        if not is_default_binding:
            super(MyInteractorStyle, obj).OnChar()

# Align cam2 by cam1
# make cam2 dist away from origin
def AlignCameraDirection(cam2, cam1, dist=4.0):
    r = np.array(cam1.GetPosition()) - np.array(cam1.GetFocalPoint())
    r = r / np.linalg.norm(r) * dist

    cam2.SetRoll(cam1.GetRoll())
    cam2.SetPosition(r)
    cam2.SetFocalPoint(0, 0, 0)
    cam2.SetViewUp(cam1.GetViewUp())
#    print(cam2.GetModelViewTransformMatrix())

def CameraFollowCallbackFunction(caller, ev):
#    print('caller\n', caller)
#    print('ev\n', ev)
    
#    rens = caller.GetRenderWindow().GetRenderers()
#    rens.InitTraversal()
#    ren1 = rens.GetNextItem()
#    ren2 = rens.GetNextItem()
#    cam1 = ren1.GetActiveCamera()
#    cam2 = ren2.GetActiveCamera()

    cam1 = CameraFollowCallbackFunction.cam1
    cam2 = CameraFollowCallbackFunction.cam2

    AlignCameraDirection(cam2, cam1)
    
    return

def Read3DImageDataFromFile(file_name, *item, **keys):
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        img_arr, img_meta = read_tiff(file_name)
    elif file_name.endswith('.ims'):
        img_arr, img_meta = read_ims(file_name, *item, **keys)
    dbg_print(3, pprint.pformat(img_meta))
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

    dbg_print(3, 'ImportImageArray(): importing image of size:',  img_arr.shape)

    # Wild guess number of channels
    if len(img_arr.shape) == 4:
        n_ch = img_arr.shape[1]
    else:
        n_ch = 1

    if (img_meta is not None) and ('imagej' in img_meta) and \
       (img_meta['imagej'] is not None) and \
       ('voxel_size_um' in img_meta['imagej']):
        if isinstance(img_meta['imagej']['voxel_size_um'], str):
            voxel_size_um = img_meta['imagej']['voxel_size_um'][1:-1]
            voxel_size_um = tuple(map(float, voxel_size_um.split(', ')))
        else:  # assume array
            voxel_size_um = img_meta['imagej']['voxel_size_um']
    else:
        voxel_size_um = (1.0, 1.0, 1.0)

    img_importer = vtkImageImport()
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

def LoadSWCTree(filepath):
    d = np.loadtxt(filepath)
    tr = (np.int32(d[:,np.array([0,6,1])]),
          np.float64(d[:, 2:6]))
    # tree format
    # (id, parent_id, type), ...
    # (x,y,z,diameter), ...
    return tr

# Split the tree in swc into linear segments (i.e. processes).
# return processes in index of tr.
# tr = LoadSWCTree(name)
def SplitSWCTree(tr):
    # Decompose tree to line objects
    # Assume tr is well and sorted and contain only one tree

    # re-label index in tr, s.t. root is 0 and all followings continued
    tr_idx = tr[0].copy()
    max_id = max(tr_idx[:,0])   # max occur node index
    n_id = tr_idx.shape[0]      # number of nodes
    # relabel array (TODO: if max_id >> n_id, we need a different algo.)
    arr_full = np.zeros(max_id+2, dtype=np.int32)
    arr_full[-1] = -1
    arr_full[tr_idx[:,0]] = np.arange(n_id, dtype=np.int32)
    tr_idx[:,0:2] = arr_full[tr_idx[:,0:2]]
    # find branch points
    n_child,_ = np.histogram(tr_idx[1:,1], bins=np.arange(n_id, dtype=np.int32))
    n_child = np.array(n_child, dtype=np.int32)
    # n_child == 0: leaf
    # n_child == 1: middle of a path or root
    # n_child >= 2: branch point
    id_bounds = np.nonzero(n_child-1)[0]
    processes = []
    for eid in id_bounds:
        # travel from leaf to branching point or root
        i = eid
        filament = [i]
        i = tr_idx[i, 1]  # parent
        while n_child[i] == 1 and i != -1:
            filament.append(i)
            i = tr_idx[i, 1]  # parent
        if i != -1:
            filament.append(i)
        processes.append(filament[::-1])

    return processes

# return a name not occur in name_set
def GetNonconflitName(prefix, name_set):
    i = 1
    while prefix in name_set:
        prefix = prefix + ".%.3d"%i
        i += 1
    return prefix

def UpdatePropertyOTFScale(obj_prop, otf_s):
    pf = obj_prop.GetScalarOpacity()
    dbg_print(3, 'UpdatePropertyOTFScale(): obj_prop.prop_conf:', obj_prop.prop_conf)
    otf_v = obj_prop.ref_prop.prop_conf['opacity_transfer_function']['AddPoint']
    
    # initialize an array of array
    # get all control point coordinates
    v = np.zeros((pf.GetSize(), 4))
    for k in range(pf.GetSize()):
        pf.GetNodeValue(k, v[k])

    if otf_s is None:  # return old otf and current setting
        return otf_v, v

    for k in range(pf.GetSize()):
        v[k][0] = otf_s * otf_v[k][0]
        pf.SetNodeValue(k, v[k])

def UpdatePropertyCTFScale(obj_prop, ctf_s):
    ctf = obj_prop.GetRGBTransferFunction()
    # get all control point coordinates
    # location (X), R, G, and B values, midpoint (0.5), and sharpness(0) values

    ctf_v = obj_prop.ref_prop.prop_conf['color_transfer_function']['AddRGBPoint']
    
    # initialize an array of array
    # get all control point coordinates
    v = np.zeros((ctf.GetSize(), 6))
    for k in range(ctf.GetSize()):
        ctf.GetNodeValue(k, v[k])

    if ctf_s is None:  # return old ctf and current setting
        return ctf_v, v

    for k in range(ctf.GetSize()):
        v[k][0] = ctf_s * ctf_v[k][0]
        ctf.SetNodeValue(k, v[k])

def GetColorScale(obj_prop):
    # guess values of colorscale for otf and ctf
    otf_v, o_v = UpdatePropertyOTFScale(obj_prop, None)
    ctf_v, c_v = UpdatePropertyCTFScale(obj_prop, None)
    return o_v[-1][0] / otf_v[-1][0], c_v[-1][0] / ctf_v[-1][0]

def SetColorScale(obj_prop, scale):
    if hasattr(scale, '__iter__'):
        otf_s = scale[0]
        ctf_s = scale[1]
    else:  # scalar
        otf_s = ctf_s = scale
    UpdatePropertyOTFScale(obj_prop, otf_s)
    UpdatePropertyCTFScale(obj_prop, ctf_s)

def ReadGUIConfigure(self, gui_conf_path):
    conf = DefaultGUIConfig()
    if os.path.isfile(gui_conf_path):
        conf_ext = json.loads(open(gui_conf_path).read())
        MergeFullDict(conf, conf_ext)
    return conf

def ReadScene(self, scene_file_path):
    scene = DefaultSceneConfig()
    if os.path.isfile(scene_file_path):
        scene_ext = json.loads(open(scene_file_path).read())
        MergeFullDict(scene, scene_ext)
    return scene

# Rotate camera
class execSmoothRotation():
    def __init__(self, cam, degree_per_sec):
        self.actor = cam
        self.degree_per_sec = degree_per_sec
        self.time_start = None
        self.time_last_update = self.time_start

    def startat(self, time_start):
        self.time_start = time_start
        self.time_last_update = self.time_start

    def __call__(self, obj, event, time_now):
        if time_now < self.time_start:
            return
        t_last_elapsed = time_now - self.time_last_update
        self.actor.Azimuth(self.degree_per_sec * t_last_elapsed)
        self.time_last_update = time_now
        iren = obj
        iren.GetRenderWindow().Render()
        #print('execSmoothRotation: Ren', time_now - self.time_start)

# Sign up to receive TimerEvent
class timerHandler():
    def __init__(self, interactor, duration, exec_obj):
        self.exec_obj = exec_obj
        self.interactor = interactor
        self.timerId = None
        self.time_start = 0
        self.duration = duration

    def callback(self, obj, event):
        t_now = time.time()
        if t_now - self.time_start > self.duration:
            self.stop()
            # align the time to the exact boundary
            t_now = self.time_start + self.duration
        self.exec_obj(obj, event, t_now)

    def start(self):
        self.interactor.AddObserver('TimerEvent', self.callback)
        self.time_start = time.time()
        self.exec_obj.startat(self.time_start)
        self.timerId = self.interactor.CreateRepeatingTimer(10)
    
    def stop(self):
        if self.timerId:
            self.interactor.DestroyTimer(self.timerId)

    def __del__(self):
        self.stop()

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
        self.GUISetup(DefaultGUIConfig())
        self.AppendToScene(DefaultSceneConfig())

    def GetNonconflitName(self, name_prefix):
        return GetNonconflitName(name_prefix, self.scene_objects.keys())

    # setup window, renderers and interactor
    def GUISetup(self, gui_conf):
        dbg_print(4, gui_conf)
        if "window" in gui_conf:
            # TODO: stop the old window?
            # TODO: try vtkVRRenderWindow?
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

        # Ref: Demonstrates the use of two renderers. Notice that the second (and subsequent) renderers will have a transparent background.
        # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
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
        interactor.SetInteractorStyle(MyInteractorStyle(interactor, self))
    #    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
        interactor.SetRenderWindow(self.render_window)
        self.interactor = interactor
        
        # first time render, for 'Timer" event to work in Windows
        self.render_window.Render()

    # The property describes how the data will look.
    def AddObjectProperty(self, name, prop_conf):
        if name in self.object_properties:
            # TODO: do we need to remove old mappers?
            pass
        dbg_print(4, 'AddObjectProperty():', name, ':', prop_conf)
        if name.startswith("volume"):
            volume_property = vtkVolumeProperty()
            
            if 'copy_from' in prop_conf:
                dbg_print(4, 'in if copy_from branch.')
                # construct a volume property by copying from exist
                ref_prop = self.object_properties[prop_conf['copy_from']]
                volume_property.DeepCopy(ref_prop)
                volume_property.prop_conf = prop_conf
                volume_property.ref_prop = ref_prop
                self.object_properties.update({name: volume_property})
                self.ModifyObjectProperty(name, prop_conf)
                return

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
                ctf_v_e = np.array(ctf_v).copy()
                for v in ctf_v_e:
                    v[0] = v[0] *  ctf_s
                # Create transfer mapping scalar value to color.
                ctf = vtkColorTransferFunction()
                for v in ctf_v_e:
                    ctf.AddRGBPoint(*v)
                volume_property.SetColor(ctf)

            volume_property.ShadeOn()

            if 'interpolation' in prop_conf:
                if prop_conf['interpolation'] == "cubic":
                    volume_property.SetInterpolationType(
                        VTK_CUBIC_INTERPOLATION)
                elif prop_conf['interpolation'] == "linear":
                    volume_property.SetInterpolationTypeToLinear()
                else:
                    dbg_print(2, "AddObjectProperty(): unknown interpolation type")
            volume_property.prop_conf = prop_conf
            object_property = volume_property
        else:
            dbg_print(2, "AddObjectProperty(): unknown object type")

        self.object_properties.update({name: object_property})

    def ModifyObjectProperty(self, name, prop_conf):
        obj_prop = self.object_properties[name]
        if name.startswith("volume"):
            if 'opacity_transfer_function' in prop_conf:
                otf_conf = prop_conf['opacity_transfer_function']
                if 'opacity_scale' in otf_conf:
                    otf_s = otf_conf['opacity_scale']
                    UpdatePropertyOTFScale(obj_prop, otf_s)
            if 'color_transfer_function' in prop_conf:
                ctf_conf = prop_conf['color_transfer_function']
                if 'trans_scale' in ctf_conf:
                    ctf_s = ctf_conf['trans_scale']
                    UpdatePropertyCTFScale(obj_prop, ctf_s)

    def AddObjects(self, name, obj_conf):
        if name in self.scene_objects:
            # TODO: do we need to remove old object?
            name = self.GetNonconflitName(name)

        renderer = self.renderers[
            obj_conf.get('renderer', '0')]

        dbg_print(3, "AddObjects: ",  obj_conf)
        dbg_print(4, "renderer: ",  obj_conf.get('renderer', '0'))

        if obj_conf['type'] == 'volume':
            # vtkVolumeMapper
            # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
            if obj_conf['mapper'] == 'GPUVolumeRayCastMapper':
                volume_mapper = vtkGPUVolumeRayCastMapper()
            elif obj_conf['mapper'] == 'FixedPointVolumeRayCastMapper':
                volume_mapper = vtkFixedPointVolumeRayCastMapper()
            else:
                # TODO: consider use vtkMultiBlockVolumeMapper
                # OR: vtkSmartVolumeMapper https://vtk.org/doc/nightly/html/classvtkSmartVolumeMapper.html#details
                # vtkOpenGLGPUVolumeRayCastMapper
                volume_mapper = vtkGPUVolumeRayCastMapper()
            #volume_mapper.SetBlendModeToComposite()

            file_path = obj_conf['file_path']
            img_importer = ImportImageFile(file_path, obj_conf)
            volume_mapper.SetInputConnection(img_importer.GetOutputPort())

            # get property used in rendering
            ref_prop_conf = obj_conf.get('property', 'volume')
            if isinstance(ref_prop_conf, dict):
                # add new property
                name = self.GetNonconflitName('volume')
                self.AddObjectProperty(name, ref_prop_conf)
                volume_property = self.object_properties[name]
            else:
                dbg_print(3, 'AddObjects(): Using existing prop:', ref_prop_conf)
                volume_property = self.object_properties[ref_prop_conf]

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

        elif obj_conf['type'] == 'swc':
            ntree = LoadSWCTree(obj_conf['file_path'])
            processes = SplitSWCTree(ntree)
            # ref: 
            # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/PolyLine/
            # https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/LinearCellDemo/
            # The procedure to add lines is:
            #    vtkPoints()  ---------------------+> vtkPolyData()
            #    vtkPolyLine() -> vtkCellArray()  /
            #   then
            #    vtkPolyData() -> vtkPolyDataMapper() -> vtkActor() -> 
            #         vtkRenderer()
            
            points = vtkPoints()
            points.SetData( numpy_to_vtk(ntree[1][:,0:3], deep=True) )
            
            cells = vtkCellArray()
            for proc in processes:
                polyLine = vtkPolyLine()
                polyLine.GetPointIds().SetNumberOfIds(len(proc))
                for i in range(0, len(proc)):
                    polyLine.GetPointIds().SetId(i, proc[i])
                cells.InsertNextCell(polyLine)

            polyData = vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(cells)

            colors = vtkNamedColors()

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polyData)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(
                colors.GetColor3d(obj_conf['color']))
            renderer.AddActor(actor)

            scene_object = actor

        elif obj_conf['type'] == 'AxesActor':
            # Create Axes object to indicate the orientation
            # vtkCubeAxesActor()
            # https://kitware.github.io/vtk-examples/site/Python/Visualization/CubeAxesActor/

            # Dynamically change position of Axes
            # https://discourse.vtk.org/t/dynamically-change-position-of-axes/691
            # Method 1
            axes = vtkAxesActor()
            axes.SetTotalLength([1.0, 1.0, 1.0])
            axes.SetAxisLabels("true"==obj_conf.get('ShowAxisLabels', "False").lower())

#            self.interactor.AddObserver('InteractionEvent',
#                CameraFollowCallbackFunction)

            CameraFollowCallbackFunction.cam1 = self.renderers['0'].GetActiveCamera()
            CameraFollowCallbackFunction.cam2 = self.renderers['1'].GetActiveCamera()

            c = self.renderers['0'].GetActiveCamera()
            c.AddObserver('ModifiedEvent',
                CameraFollowCallbackFunction)

            renderer.AddActor(axes)
            scene_object = axes

        elif obj_conf['type'] == 'OrientationMarker':
            # Method 2
            # Ref: https://kitware.github.io/vtk-examples/site/Python/Interaction/CallBack/
            axes = vtkAxesActor()
            axes.SetTotalLength([1.0, 1.0, 1.0])
            #axes.SetAxisLabels("true"==obj_conf.get('ShowAxisLabels', "False").lower())
            axes.SetAxisLabels(True)

            # Ref: https://vtk.org/doc/nightly/html/classvtkOrientationMarkerWidget.html
            om = vtkOrientationMarkerWidget()
            om.SetOrientationMarker(axes)
            om.SetInteractor(self.interactor)
            om.SetDefaultRenderer(renderer)
            om.EnabledOn()
            om.SetInteractive(False)
            #om.InteractiveOn()
            om.SetViewport(0, 0, 0.2, 0.2)
            # TODO: the vtkOrientationMarkerWidget and timerHandler can cause program lose respons or Segmentation fault, for unknown reason.

            scene_object = om

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
            obj_desc = {'filepath': obj_desc}
        
        if 'filepath' in obj_desc:
            file_path = obj_desc['filepath']
            if file_path.endswith('.tif'):
                # assume this a volume
                obj_conf = {
                    "type": "volume",
                    "mapper": "GPUVolumeRayCastMapper",
                    "view_point": "auto",
                    "file_path": file_path
                }
                name = self.GetNonconflitName('volume')
                self.AddObjects(name, obj_conf)
            elif file_path.endswith('.ims') or file_path.endswith('.h5'):
                # assume this a IMS volume
                obj_conf = {
                    "type": "volume",
                    "mapper": "GPUVolumeRayCastMapper",
                    "view_point": "auto",
                    "file_path": file_path,
                    "level": obj_desc.get('level', '0'),
                    "channel": obj_desc.get('channel', '0'),
                    "time_point": obj_desc.get('time_point', '0'),
                    "range": obj_desc.get('range', '[:,:,:]')
                }
                if 'colorscale' in obj_desc:
                    s = float(obj_desc['colorscale'])
                    obj_conf.update({'property': {
                        'copy_from': 'volume',
                        'opacity_transfer_function': {'opacity_scale': s},
                        'color_transfer_function'  : {'trans_scale': s}
                    }})
                name = self.GetNonconflitName('volume')
                self.AddObjects(name, obj_conf)
            else:
                dbg_print(1, "Unreconized source format.")
            
        if 'swc' in obj_desc:
            name = self.GetNonconflitName('swc')
            obj_conf = {
                "type": 'swc',
                "color": obj_desc.get('fibercolor','Tomato'),
                "file_path": obj_desc['swc']
            }
            self.AddObjects(name, obj_conf)

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
    parser.add_argument('--filepath', help='image stack filepath')
    parser.add_argument('--level', help='for multi-level image (.ims), load only that level')
    parser.add_argument('--channel', help='Select channel for IMS image.')
    parser.add_argument('--time_point', help='Select time point for IMS image.')
    parser.add_argument('--range', help='Select range within image.')
    parser.add_argument('--colorscale', help='Set scale of color transfer function.')
    parser.add_argument('--swc', help='Read and draw swc file.')
    parser.add_argument('--fibercolor', help='Set fiber color.')
    args = parser.parse_args()
    # convert class attributes to dict
    keys = ['filepath', 'level', 'channel', 'time_point', 'range',
            'colorscale', 'swc', 'fibercolor']
    d = {k: getattr(args, k) for k in keys
            if hasattr(args, k) and getattr(args, k)}
    dbg_print(3, 'get_program_parameters(): d=', d)
    return d

if __name__ == '__main__':
    gui = GUIControl()
    gui.EasyObjectImporter(get_program_parameters())
    gui.Start()
    
