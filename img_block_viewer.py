#!/usr/bin/env python3
# A simple viewer based on PyVTK for volumetric data and neuronal SWC data,
# specialized for viewing of neuron tracing results.

# Dependencies:
# pip install vtk opencv-python tifffile h5py scipy joblib IPython

# Usage examples:
# python img_block_viewer.py --filepath RM006_s128_c13_f8906-9056.tif
# ./img_block_viewer.py --filepath z00060_c3_2.ims --level 3 --range '[400:800, 200:600, 300:700]' --colorscale 10
# ./img_block_viewer.py --filepath 3864-3596-2992_C3.ims --colorscale 10 --swc R2-N1-A2.json.swc_modified.swc --fibercolor green
# ./img_block_viewer.py --lychnis_blocks RM006-004-lychnis/image/blocks.json --swc RM006-004-lychnis/F5.json.swc
# ./img_block_viewer.py --scene scene_example_vol_swc.json
# ./img_block_viewer.py --scene scene_example_rm006_3.json --lychnis_blocks RM006-004-lychnis/image/blocks.json

# See help message for more tips.
# ./img_block_viewer.py -h

# Program logic:
#   In a very general sense, this code does the following:
#     * Read the window configuration and scene object description.
#     * Load the image or SWC data according to the description.
#     * Pass the data to VTK for rendering.
#     * Let VTK to handle the GUI interaction.
#   Essentially this code translate the object description to VTK commands
#   and does the image data loading.

# Code structure:
#   utils.py            : General utilizer functions,
#                         and VTK related utilizer functions.
#   data_loader.py      : Image and SWC loaders.
#   ui_interactions.py  : Keyboard and mouse interaction.
#   cg_translators.py   : translate json(dict) style object description to 
#                         vtk commands; also represent high level objects.
#   img_block_viewer.py : GUI control class
#                         Loads window settings, object properties, objects.
#                         Command line related data import function.

# Performance tip:
# For memory footprint:
# n_neuron = 1660 (SWC), n_points = 39382068 (0.44 GiB)
# float32 mode:
# RAM = 2.4GiB (5.3GiB during pick), 3.2g(after pick),
# GPU = 1128MiB
# float64 mode:
# RAM = 3.3GiB (8.3GiB during pick), 3.6g(after pick),
# GPU = 1128MiB

# Performance for time:
# load 6356 neurons: <19m34.236s

import os
import os.path
import time
import argparse
import json
import joblib

import numpy as np
from numpy import array as _a

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2
import vtkmodules.vtkRenderingFreeType

from vtkmodules.vtkCommonCore import (
    vtkPoints,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
    vtkColorSeries
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine,
    vtkPlane,
)
from vtkmodules.vtkIOImage import (
    vtkPNGWriter,
)
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter,
    vtkPropPicker,
    vtkPointPicker,
    vtkActor,
    vtkPolyDataMapper,
    vtkTextActor,
)
# 
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from vtkmodules.vtkFiltersCore import vtkClipPolyData

from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette

# loading this consumes ~0.1 second!
# might move it to where it is used.
from vtkmodules.util.numpy_support import numpy_to_vtk

#import vtkmodules.all as vtk

import utils
from utils import (
    GetNonconflitName,
    dbg_print,
    WindowsFriendlyDateTime,
)
utils.debug_level = 5

from ui_interactions import (
    GenerateKeyBindingDoc,
    PointSetHolder,
)
from data_loader import OnDemandVolumeLoader
from cg_translators import ObjTranslator

# Default configuration parameters are as follow
def DefaultGUIConfig():
    d = {
        "window": {
            "size": "auto",
            "title": "SimpleRayCast",
            "number_of_layers": 2,
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
                "type": "volume",
                "opacity_transfer_function": {
                    "AddPoint": [
                        [20, 0.1],
                        [255, 0.9]
                    ],
                    "opacity_scale": 1.0
                },
                "color_transfer_function": {
                    "AddRGBPoint": [
                        [0.0, 0.0, 0.0, 0.0],
                        [64.0, 0.0, 0.2, 0.0],
                        [128.0, 0.0, 0.7, 0.1],
                        [255.0, 0.0, 1.0, 0.2]
                    ],
                    "trans_scale": 1.0
                },
                "interpolation": "cubic"
            },
            "volume_default_composite": {
                "type": "volume",
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
                "color": "Black"
            },
            "3d_cursor": {
                "type": "Sphere",
            },
            "camera1": {
                "type": "Camera",
                "renderer": "0",
                "Azimuth": 45,
                "Elevation": 30,
                "clipping_range": [0.0001, 100000]
            },
            "camera2": {
                "type": "Camera",
                "renderer": "1",
                "follow_direction": "camera1"
            },
            "orientation_axes": {
                "type": "AxesActor",
                "ShowAxisLabels": False,
                "renderer": "1",
            },
        }
    }
    return d

def ReadGUIConfigure(gui_conf_path):
    conf = DefaultGUIConfig()
    if os.path.isfile(gui_conf_path):
        conf_ext = json.loads(open(gui_conf_path).read())
        MergeFullDict(conf, conf_ext)
    return conf

def ReadScene(scene_file_path):
    scene = DefaultSceneConfig()
    if os.path.isfile(scene_file_path):
        scene_ext = json.loads(open(scene_file_path).read())
        MergeFullDict(scene, scene_ext)
    return scene

def ShotScreen(render_window, filename):
    """
    Take a screenshot.
    Save to filename
    """
    # From: https://kitware.github.io/vtk-examples/site/Python/Utilities/Screenshot/
    win2if = vtkWindowToImageFilter()
    win2if.SetInput(render_window)
    win2if.SetInputBufferTypeToRGB()
    win2if.ReadFrontBufferOff()
    win2if.Update()

    # If need transparency in a screenshot
    # https://stackoverflow.com/questions/34789933/vtk-setting-transparent-renderer-background
    
    writer = vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(win2if.GetOutputPort())
    writer.Write()

class VolumeClipper:
    """
    Function: Cut the volume with a box surrounding the points, 
              which is represented by 6 mutually perpendicular planes.
    Usage   : Initialize this class, use 'SetPoints()' to set the points
              to be surrounded, and call the 'CutVolume()' function.
    """
    def __init__(self, points, box_scaling=1, min_boundary_length=10):
        """
        Parameter description:
          points               : the Points to calculate the bounding box
          box_scaling          : the scale of the bouding box
          min_boundary_length  : the min length/width/height of the bounding box 
        """
        self.points = None
        self.planes = None
        self.box_scaling = box_scaling
        self.min_boundary_length = min_boundary_length
        self.SetPoints(points)
    
    def CreatePlane(self, origin, normal):
        p = vtkPlane()
        p.SetOrigin(origin)
        p.SetNormal(normal)
        return p
    
    def Get6SurroundingPlanes(self, points, box_scaling = 1,
                              min_boundary_length = 10):
        """
        Calculate the bounding box and express it in plane form
        Parameter description:
          points              : the points to calculate the bounding box
          box_scaling         : the scale of the bounding box
          min_boundary_length : the min length/width/height of the bounding box
        """

        center_point = points.mean(axis=0)
        # Use center_point as the origin and calculate the coordinates of points
        subtracted = points - center_point
        # Calculate basis vectors
        uu, dd, V = np.linalg.svd(subtracted)
        # The natural basis of the point set
        basis_vectors = V
        # Calculate the projection length of the points on the basis vectors
        projection_length = subtracted @ basis_vectors.T
        # The length, width and height of the box 
        #  in the direction of the basis vectors
        box_LWH_basis = np.ptp(projection_length, axis=0)
        # The box center coordinate with respect to basis vectors, 
        #  using the center_point as the origin
        box_center_basis = np.min(projection_length, axis=0) + \
                           box_LWH_basis / 2
        # Convert the coordinate system back
        box_center = center_point + box_center_basis @ basis_vectors
        # Set the minimum length/width/height of the box  
        box_LWH_basis[ np.where(box_LWH_basis < min_boundary_length) ] = \
            min_boundary_length
        # Generate planes
        plane_normals = np.vstack((basis_vectors, -basis_vectors))
        planes = [
            self.CreatePlane(
                box_center \
                - (box_scaling * box_LWH_basis[i%3]/2 + min_boundary_length) \
                   * plane_normals[i],
                plane_normals[i]
            )
            for i in range(plane_normals.shape[0])
        ]
        return planes

    def SetPoints(self, points):
        """ Set the points to be surrounded. """
        # TODO: should we remove the planes first?
        self.points = points
        self.planes = self.Get6SurroundingPlanes(points)

    def CutVolume(self, volume):
        """ Add clipping planes to the mapper of the volume. """
        m = volume.GetMapper()
        for each_plane in self.planes:
            m.AddClippingPlane(each_plane)

    def CutVolumes(self, volumes):
        volumes.InitTraversal()
        v = volumes.GetNextVolume()
        while v is not None:
            self.CutVolume(v)
            v = volumes.GetNextVolume()

    @staticmethod
    def RestoreVolume(volume):
        """ Remove all the clipping planes attached to the volume. """
        m = volume.GetMapper()
        # Remove all the clipping planes
        m.RemoveAllClippingPlanes()

    @staticmethod
    def RestoreVolumes(volumes):
        """ Remove all the clipping planes for all the volume in the scene. """
        volumes.InitTraversal()
        v=volumes.GetNextVolume()
        while v is not None:
            VolumeClipper.RestoreVolume(v)
            v = volumes.GetNextVolume()

class PointSearcher:
    """
    For a given point coordinate and connectivity graph,
    search connected nearby points.
    """

    def __init__(self, point_graph, level = 5, points_coor = None):
        self.point_graph = point_graph
        self.visited_points = set()
        self.level = level
        self.points_coordinate = points_coor

    def SetTargetPoint(self, target_point):
        self.visited_points = set()
        self.target = target_point

    def SetPointGraph(self, point_graph):
        self.visited_points = set()
        self.point_graph = point_graph

    def SetNumberOfSearchLayers(self, number):
        self.visited_points = set()
        self.level = number

    def DFS(self, pid, level):
        if pid == -1 or pid in self.visited_points:
            return
        if level > 0:
            self.visited_points.add(pid)
            for each in self.point_graph[pid].indices:
                self.DFS(each, level - 1)

    def DFS_path(self, pid, level, path):
        if pid == -1 or pid in self.visited_points:
            return
        if level > 0:
            self.visited_points.add(pid)
            for each in self.point_graph[pid].indices:
                if each in self.visited_points:
                    continue
                path.append([pid, each])
                self.DFS_path(each, level - 1, path)

    def SearchPathAround(self, pid):
        self.visited_points = set()
        path = []
        self.DFS_path(pid, self.level * 2, path)
        return list(self.visited_points), path

    def SearchPointsAround(self, pid):
        self.visited_points = set()
        self.DFS(pid, self.level)
        return list(self.visited_points)

    def SearchPointsAround_coor(self, pid):
        coor = self.points_coordinate[:, self.SearchPointsAround(pid)]
        return coor.T

class FocusModeController:
    """
    This class manages the focus mode and is mainly responsible for cutting blocks and lines
    """
    def __init__(self):
        self.gui_ctrl = None
        self.renderer = None
        self.iren = None
        self.point_searcher = None
        self.center_point = None
        self.volume_clipper = None
        self.isOn = False
        self.swc_polydata = None
        self.swc_mapper = None
        self.cut_swc_flag = True
        self.focus_swc = None

    def SetPointsInfo(self, point_graph, point_coor):
        self.point_searcher = PointSearcher(point_graph, points_coor=point_coor)

    def SetGUIController(self, gui_ctrl):
        self.gui_ctrl = gui_ctrl
        self.renderer = self.gui_ctrl.GetMainRenderer()
        self.iren = gui_ctrl.interactor
        self.gui_ctrl.volume_observers.append(self)
        swc_objs = self.gui_ctrl.GetObjectsByType('swc', 1)
        if swc_objs:
            self.swc_mapper = swc_objs[0].actor.GetMapper()
            self.swc_polydata = self.swc_mapper.GetInput()
        else:
            self.swc_mapper = None
            self.swc_polydata = None
        self.point_searcher = PointSearcher(
            self.gui_ctrl.point_graph,
            points_coor = self.gui_ctrl.point_set_holder())

    def SetCenterPoint(self, pid):
        """
        Update spot site according to point position.
        """
        self.center_point = pid
        if self.isOn:
            points = self.point_searcher \
                     .SearchPointsAround_coor(self.center_point)
            if not self.volume_clipper:
                self.volume_clipper = VolumeClipper(points)
            else:
                self.volume_clipper.SetPoints(points)
            self.gui_ctrl.UpdateVolumesNear(
                self.point_searcher.points_coordinate.T[self.center_point])
            self.volume_clipper.RestoreVolumes(self.renderer.GetVolumes())
            self.volume_clipper.CutVolumes(self.renderer.GetVolumes())
            if self.cut_swc_flag:
                if self.focus_swc:
                    self.gui_ctrl.GetMainRenderer().RemoveActor(self.focus_swc)
                oldClipper = vtkClipPolyData()
                oldClipper.SetInputData(self.swc_polydata)
                oldClipper.SetClipFunction(self.volume_clipper.planes[0])
                path = self.point_searcher.SearchPathAround(self.center_point)
                self.swc_mapper.SetInputData(oldClipper.GetOutput())
                self.CreateLines(path[1])
            self.iren.GetRenderWindow().Render()

    def Toggle(self):
        if self.isOn:
            self.isOn = False
            self.volume_clipper.RestoreVolumes(self.renderer.GetVolumes())
            if self.cut_swc_flag:
                self.swc_mapper.SetInputData(self.swc_polydata)
                self.gui_ctrl.GetMainRenderer().RemoveActor(self.focus_swc)
            self.iren.GetRenderWindow().Render()
        else:
            self.isOn = True
            if self.center_point:
                self.SetCenterPoint(self.center_point)

    def CreateLines(self, path):
        points = vtkPoints()
        points.SetData(numpy_to_vtk(
            self.gui_ctrl.point_set_holder().T, deep=True))
        cells = vtkCellArray()
        for proc in path:
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
            colors.GetColor3d('yellow'))
        self.gui_ctrl.GetMainRenderer().AddActor(actor)
        self.focus_swc = actor

    def Notify(self, volume):
        if self.isOn:
            self.volume_clipper.CutVolume(volume)

class GUIControl:
    """
    Controller of VTK.
    Interact with VTK directly to setup the scene and the interaction.
    """
    def __init__(self):
        # Load configure
        self.renderers = {}
        self.render_window = None
        self.interactor = None
        self.translator = ObjTranslator()
        self.object_properties = {}
        self.scene_objects = {}
        self.selected_objects = []
        self.main_renderer_name = None
        self.do_not_start_interaction = False
        self.li_cg_conf = []
        
        self.utility_objects = {}
        self.volume_loader = OnDemandVolumeLoader()
        
        self.win_conf = {}
        self.scene_saved = {
            'object_properties': {},
            'objects': {}
        }
        self.point_set_holder = PointSetHolder()
        # The point graph is initialized when adding SWC type objects and used to find adjacent points
        self.point_graph = None
        # If a new volume is loaded, it will be clipped by the observers
        self.volume_observers = []
        self.selected_pid = None
        self.focusController = FocusModeController()

        # load default settings
        self.loading_default_config = True
        self.GUISetup(DefaultGUIConfig())
        self.AppendToScene(DefaultSceneConfig())
        self.loading_default_config = False
        self.n_max_cpu_cores_default = 8
        self.swc_loading_batch_size = 2

    def GetNonconflitName(self, name_prefix, name_book = 'scene'):
        if name_book == 'scene':
            index = self.scene_objects
        elif name_book == 'property':
            index = self.object_properties
        return GetNonconflitName(name_prefix, index.keys())

    def GetMainRenderer(self):
        if self.main_renderer_name:
            return self.renderers[self.main_renderer_name]
        return None

    def UtilizerInit(self):
        colors = vtkNamedColors()
        
        silhouette = vtkPolyDataSilhouette()
        silhouette.SetCamera(self.GetMainRenderer().GetActiveCamera())

        # Create mapper and actor for silhouette
        silhouetteMapper = vtkPolyDataMapper()
        silhouetteMapper.SetInputConnection(silhouette.GetOutputPort())

        silhouetteActor = vtkActor()
        silhouetteActor.SetMapper(silhouetteMapper)
        silhouetteActor.GetProperty().SetColor(colors.GetColor3d('Tomato'))
        silhouetteActor.GetProperty().SetLineWidth(5)

        self.utility_objects['silhouette'] = [silhouette, silhouetteActor]

    def Set3DCursor(self, xyz):
        # operate on object: 3d_cursor
        if '3d_cursor' in self.scene_objects:
            dbg_print(4, 'Set 3D cursor to', xyz)
            cursor = self.scene_objects['3d_cursor']
            cursor.position = xyz
            self.render_window.Render()

    def SetSelectedPID(self, pid):
        if pid < len(self.point_set_holder):
            self.selected_pid = pid
            self.focusController.SetCenterPoint(pid)
            self.Set3DCursor(self.point_set_holder()[:,pid])
        else:
            dbg_print(3, 'SetSelectedPID(): pid=%d out of range, len=%d' \
                      %(pid, len(self.point_set_holder)))

    def GetObjectsByType(self, str_type, n_find = -1):
        """ Find all (at most n_find) objects with the type str_type. """
        li_obj = []
        for k, conf in self.scene_saved['objects'].items():
            if conf['type'] == str_type:
                li_obj.append(self.scene_objects[k])
                n_find -= 1
                if n_find == 0:
                    break
        return li_obj

    def Get3DCursor(self):
        cursor = self.scene_objects.get('3d_cursor', None)
        if cursor:
            center = cursor.position
        else:
            center = None
            dbg_print(2, 'Get3DCursor(): no 3d coor found.')
        return center

    def GUISetup(self, gui_conf):
        """ setup window, renderers and interactor """
        dbg_print(4, gui_conf)
        if 'window' in gui_conf:
            win_conf = gui_conf['window']
            self.translator.init_window(self, None).parse(win_conf)

        if 'renderers' in gui_conf:
            self.translator.init_renderers(self, self.renderers) \
                            .parse(gui_conf['renderers'])

        # first one is the main
        self.main_renderer_name = \
            next(iter(self.renderers.keys()))

        if 'window' in gui_conf:
            win_conf = gui_conf['window']
            self.translator.init_window(self, self.renderers) \
                            .parse_post_renderers(win_conf)

        self.interactor = self.translator \
                .init_interactor(self, self.renderers) \
                .parse(None)

        # first time render, for 'Timer' event to work in Windows
        self.render_window.Render()

    def WindowConfUpdate(self, new_win_conf):
        dbg_print(4, 'WindowConfUpdate():', new_win_conf)
        self.translator.init_window(self, None).parse(new_win_conf)
        if 'full_screen' in new_win_conf:
            # after switch between full screen/window mode,
            # we need this to let the keyboard/mouse interaction to work
            self.interactor.ReInitialize()   
        self.win_conf.update(new_win_conf)

    # The property describes how the data will look.
    def AddObjectProperty(self, name, prop_conf):
        if name in self.object_properties:
            # TODO: do we need to remove old mappers?
            dbg_print(2, 'AddObjectProperty(): conflict name: ', name)
            dbg_print(2, '                     will be overwritten.')
        dbg_print(3, 'AddObjectProperty(): "'+name+'" :', prop_conf)
        if name.startswith('volume'):
            object_property = self.translator \
                            .translate_prop_conf(self, prop_conf)
        else:
            dbg_print(2, 'AddObjectProperty(): unknown object type')

        if not self.loading_default_config:
            self.scene_saved['object_properties'][name] = prop_conf

        self.object_properties.update({name: object_property})

    def AddObject(self, name, obj_conf):
        if name in self.scene_objects:
            if (obj_conf['type'] == 'Camera') and \
                obj_conf.get('new', False) == False:
                # Special rule for camera: we mostly update (modify) camera.
                # in this case, we do not update the object name.
                dbg_print(4, 'AddObject(): update Camera.')
            else:
                # new object desired
                dbg_print(2, 'AddObject(): conflict name: ', name)
                name = self.GetNonconflitName(name)
                dbg_print(2, '             renamed to: ', name)

        renderer = self.renderers[
            obj_conf.get('renderer', '0')]

        dbg_print(3, 'AddObject: "' + name + '" :', obj_conf)
        dbg_print(4, 'renderer: ',  obj_conf.get('renderer', '0'))

        scene_object = self.translator \
                       .translate_obj_conf(self, renderer, obj_conf)

        if obj_conf['type'] == 'volume':
            self.selected_objects = [name]

        if obj_conf['type'] == 'swc':
            self.point_set_holder.AddPoints(scene_object.PopRawPoints(), name)

        if not self.loading_default_config:
            if name in self.scene_saved['objects']:
                self.scene_saved['objects'][name].update(obj_conf)
            else:
                self.scene_saved['objects'][name] = obj_conf
        
        self.scene_objects.update({name: scene_object})

    def GetNonconflitNameBatch(self, name_prefix, n):
        name_similar = [name for name in self.scene_objects.keys() \
                        if name.startswith(name_prefix)]
        name_similar.sort()
        name_k = [int(k.split('.')[1]) for k in name_similar \
                  if (len(k.split('.')) == 2) and 
                     (k.split('.')[1].isdecimal())]
        if len(name_k) == 0:
            i0 = 1
        else:
            i0 = max(name_k) + 1
        idx = np.arange(i0, i0 + n)
        li_name = [name_prefix + '.%.d'%k for k in idx]
        return li_name

    def AddBatchSWC(self, name_prefix, li_swc_conf):
        """ add swc in parallel """

        # get object names
        li_name = self.GetNonconflitNameBatch(name_prefix, len(li_swc_conf))

        # get swc data paths
        li_file_path = [o['file_path'] for o in li_swc_conf]

        # split the workload into batches, each batch contains multiple jobs
        n_batch_size = self.swc_loading_batch_size
        li_file_path_batch = []
        k = 0
        while k < len(li_file_path):
            li_file_path_batch.append(li_file_path[k : k + n_batch_size])
            k += n_batch_size
        dbg_print(5, f'AddBatchSWC(): n_jobs = {len(li_file_path)}, batch_size = {n_batch_size}, n_batch = {len(li_file_path_batch)}')
        n_job_cores = min(self.n_max_cpu_cores_default, joblib.cpu_count())
        dbg_print(4, f'AddBatchSWC(): using {n_job_cores} cores')

        def batch_load(file_path_batch):
            dbg_print(5, '...dealing', file_path_batch)
            results = []
            for f in file_path_batch:
                #results.append(self.translator.obj_swc.LoadRawSwc(f))
                results.append(ObjTranslator.obj_swc.LoadRawSwc(f))
            return results

        dbg_print(4, 'AddBatchSWC(): loading...')
        # load swc data in parallel
        #cached_pointsets = [batch_load(j)
        #                    for j in li_file_path_batch]
        # TODO: the parallel execution has a large fixed overhead,
        #       this overhead is not sensitive to batch size,
        #       which indicate the overhead is probably related to
        #       large data transfer (RAM IO?).
        #       The profiler (cProfile) sees
        #       {method 'acquire' of '_thread.lock' objects} which consumes
        #       most of the run time.
        t1 = time.time()
        cached_pointsets = joblib.Parallel(n_jobs = n_job_cores) \
                (joblib.delayed(batch_load)(j) for j in li_file_path_batch)
        t2 = time.time()
        dbg_print(4, f'                       done, t = {t2-t1:.3f} sec.')

        # unpack the batch results
        cached_pointsets_expanded = []
        for b in cached_pointsets:
            cached_pointsets_expanded.extend(b)
        cached_pointsets = cached_pointsets_expanded

        # initialize translator units
        ren = self.GetMainRenderer()
        li_tl = [self.translator.obj_swc(self, ren) \
                  for k in range(len(li_swc_conf))]

        t3 = time.time()
        dbg_print(4, f'AddBatchSWC(): init obj_swc, t = {t3-t2:.3f} sec.')

        dbg_print(4, 'AddBatchSWC(): parsing...')
        # parse the swc_conf
        for k in range(len(li_tl)):
            name         = li_name[k]
            swc_conf     = li_swc_conf[k]
            li_tl[k].cache1 = tuple(cached_pointsets[k])
            scene_object = li_tl[k].parse(swc_conf)
            self.point_set_holder.AddPoints(scene_object.PopRawPoints(), name)
            self.scene_objects.update({name: scene_object})
            self.scene_saved['objects'].update({name: swc_conf})
        t4 = time.time()
        dbg_print(4, f'                       done, t = {t4-t3:.3f} sec.')

    def AppendToScene(self, scene_conf):
        """ add objects to the renderers """
        if 'object_properties' in scene_conf:
            for key, prop_conf in scene_conf['object_properties'].items():
                self.AddObjectProperty(key, prop_conf)

        if 'objects' in scene_conf:
            for key, obj_conf in scene_conf['objects'].items():
                self.AddObject(key, obj_conf)
        # see also vtkAssembly
        # https://vtk.org/doc/nightly/html/classvtkAssembly.html#details
        return

    def RemoveObject(self, name):
        if name not in self.scene_objects:
            dbg_print(2,'RemoveObject(): object non-exist:', name)
            return
        dbg_print(3, 'Removing object:', name)
        # TODO: Do not remove if it is an active camera
        self.scene_objects[name].remove()
        
        if name in self.selected_objects:
            self.selected_objects.remove(name)
        del self.scene_objects[name]
        del self.scene_saved['objects'][name]
        # TODO: correctly remove a object, possibly from adding process.

    def LoadVolumeNear(self, pos, radius=20):
        if (pos is None) or (len(pos) != 3):
            return []
        vol_list = self.volume_loader.LoadVolumeAt(pos, radius)
        #print(vol_list)
        dbg_print(3, 'LoadVolumeNear(): n_loaded =', len(vol_list))
        get_vol_name = lambda p: os.path.splitext(os.path.basename(p))[0]
        for v in vol_list:
            v_name = 'volume' + get_vol_name(v['image_path'])
            if v_name in self.scene_objects:  # skip repeated block
                continue
            self.AddObject(
                v_name,
                {
                    'type'      : 'volume',
                    'file_path' : v['image_path'],
                    'origin'    : v['origin'],
                    'view_point': 'keep'
                },
            )
        return vol_list

    def UpdateVolumesNear(self, point_pos, radius = 20):
        focus_vols = self.volume_loader.LoadVolumeAt(point_pos, radius)
        scene_vols = self.scene_objects.copy()
        get_vol_name = lambda p: os.path.splitext(os.path.basename(p))[0]
        focus_vols_name_set = set()
        for v in focus_vols:
            focus_vols_name_set.add('volume' + get_vol_name(v['image_path']))
        add_set = []
        for vol in focus_vols:
            name = 'volume' + get_vol_name(vol['image_path'])
            if name not in scene_vols:
                add_set.append(vol)
        for sv in scene_vols:
            if sv not in focus_vols_name_set and type(scene_vols[sv]) == self.translator.obj_volume:
                if sv is not self.selected_objects[0]:
                    self.RemoveObject(sv)
        for each in add_set:
            self.AddObject(
                'volume' + get_vol_name(each['image_path']),
                {
                    'type': 'volume',
                    'file_path': each['image_path'],
                    'origin': each['origin'],
                    'view_point': 'keep'
                },
            )

    def ShowOnScreenHelp(self):
        if not hasattr(self, 'text_actor'):
            win_size = self.render_window.GetSize()

            font_size = int(31 * win_size[1] / 1600)
            text_actor = vtkTextActor()
            text_actor.SetInput(GenerateKeyBindingDoc())
            prop = text_actor.GetTextProperty()
            prop.SetFontFamilyToCourier()
            prop.SetColor(1.0, 1.0, 0.0)
            prop.SetBackgroundColor(0.0, 0.2, 0.5)
            prop.SetBackgroundOpacity(0.8)
            prop.FrameOn()
            prop.SetFontSize(font_size)
            # put the text box to center of window
            prop.SetVerticalJustificationToCentered()
            v = _a([0]*4)
            text_actor.GetBoundingBox(self.GetMainRenderer(), v)
            text_actor.SetPosition(win_size[0]/2 - (v[1]-v[0])/2, win_size[1]/2)

            self.text_actor = text_actor
            self.GetMainRenderer().AddActor2D(text_actor)
        else:
            # remove the text actor
            self.GetMainRenderer().RemoveActor(self.text_actor)
            del self.text_actor
        self.render_window.Render()

    def EasyObjectImporter(self, cmd_obj_desc):
        """
        Used to accept command line inputs which need default parameters.
        """
        if not cmd_obj_desc:
            return
        if isinstance(cmd_obj_desc, str):
            cmd_obj_desc = {'filepath': cmd_obj_desc}

        self.li_cg_conf = self.translator \
                          .parse_all_cmd_args_obj(cmd_obj_desc, 'animation')

        tl_win = self.translator.init_window(self, None)
        win_conf = tl_win.parse_cmd_args(cmd_obj_desc)
        tl_win.parse(win_conf)
        tl_win.parse_post_renderers(win_conf)
        
        # so far, init_renderers, init_interactor do not have cmd options.
        
        scene_ext = self.translator.init_scene \
                    .parse_cmd_args(cmd_obj_desc)
        self.AppendToScene(scene_ext)

        li_obj_conf = self.translator.parse_all_cmd_args_obj(cmd_obj_desc)
        
        # for load SWC in parallel
        li_swc_conf = [o for o in li_obj_conf if o['type'] == 'swc']
        li_obj_conf = [o for o in li_obj_conf if o['type'] != 'swc']

        self.AddBatchSWC('swc', li_swc_conf)
        
        for obj_conf in li_obj_conf:
            name = self.GetNonconflitName(obj_conf['type'])
            self.AddObject(name, obj_conf)

    def ShotScreen(self, filename = ''):
        if filename == '':
            filename = 'screenshot_' + WindowsFriendlyDateTime() + '.png'
        ShotScreen(self.render_window, filename)

    def ExportSceneFile(self):
        # export camera data
        cam = self.GetMainRenderer().GetActiveCamera()
        c = {
            "camera1": {
                "type": "Camera",
                "renderer": self.main_renderer_name,
                "clipping_range": cam.GetClippingRange(),
                "Position"  : cam.GetPosition(),
                "FocalPoint": cam.GetFocalPoint(),
                "ViewUp"    : cam.GetViewUp(),
                "ViewAngle" : cam.GetViewAngle()
            },
        }
        self.scene_saved['objects'].update(c)
        # export scene_saved
        dbg_print(3, 'Saving scene file.')
        with open('scene_saved.json', 'w') as f:
            json.dump(self.scene_saved, f, indent=4, ensure_ascii = False)

    def Start(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.UtilizerInit()
        self.focusController.SetGUIController(self)

        for cg_conf in self.li_cg_conf:
            self.translator.translate(self, self.GetMainRenderer(),
                                      'animation_', cg_conf)
        
        if not self.do_not_start_interaction:
            self.GetMainRenderer().ResetCamera()
            #self.GetMainRenderer().ResetCameraClippingRange()
            self.interactor.Start()

def get_program_parameters():
    description = 'Simple volumetric image and SWC viewer based on PyVTK.'
    epilogue = GenerateKeyBindingDoc()
    parser = argparse.ArgumentParser(
                description=description,
                epilog=epilogue,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    ObjTranslator().add_all_arguments_to(parser)
    parser.add_argument(
        '--verbosity', type=int, choices=[0, 1, 2, 3, 4, 5],
        help="output verbosity(0(nothing), 1(error) ~ 5(verbose))")
    args = parser.parse_args()
    if args.verbosity is not None:
        utils.debug_level = args.verbosity
    # convert args from Namespace to dict, and filter out None values.
    args = {k:v for k,v in vars(args).items() if v is not None}
    dbg_print(3, 'get_program_parameters(): d=', args)
    return args

if __name__ == '__main__':
    cmd_obj_desc = get_program_parameters()
    gui = GUIControl()
    gui.EasyObjectImporter(cmd_obj_desc)
    gui.Start()

