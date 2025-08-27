#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

# A simple viewer based on PyVTK for volumetric data and neuronal SWC data,
# specialized for viewing of neuron tracing results.

# Dependencies:
# pip install vtk tifffile h5py scipy IPython

# Usage examples:
# python img_block_viewer.py --img_path RM006_s128_c13_f8906-9056.tif
# ./img_block_viewer.py --img_path z00060_c3_2.ims --level 3 --range '[400:800, 200:600, 300:700]' --colorscale 10
# ./img_block_viewer.py --img_path 3864-3596-2992_C3.ims --colorscale 10 --swc R2-N1-A2.json.swc_modified.swc --fibercolor green
# ./img_block_viewer.py --lychnis_blocks RM006-004-lychnis/image/blocks.json --swc RM006-004-lychnis/F5.json.swc
# ./img_block_viewer.py --scene scene_example_vol_swc.json
# ./img_block_viewer.py --scene scene_example_rm006_3.json --lychnis_blocks RM006-004-lychnis/image/blocks.json

# See help message for more tips.
# ./img_block_viewer.py -h

# Program logic:
#   Essentially this code loads images and SWC files, then translate 
#   the object description (in .json or python dict()) to VTK commands.
#   i.e.
#     * Read the window configuration and scene object description.
#     * Load the image or SWC data according to the description.
#     * Pass the data to VTK for rendering.
#     * Let VTK to handle the GUI interaction.

# Code structure:
#   utils.py            : General utilizer functions,
#                         and VTK related utilizer functions.
#   data_loader.py      : Image and SWC loaders.
#   cg_operations.py    : Procedures related to operating CG objects.
#   ui_interactions.py  : Keyboard and mouse interaction.
#   cg_translators.py   : translate json(dict) style object description to 
#                         vtk commands; also represent high level objects.
#   img_block_viewer.py : GUI control class
#                         Loads window settings, object properties, objects.
#                         Command line related data import function.

import time
g_t0 = time.time()
import os
import os.path
import argparse
import json
import copy

from multiprocessing import Pool
from multiprocessing import cpu_count
# Fix for pyinstaller
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
from multiprocessing import freeze_support

import numpy as np
from numpy import array as _a

# TODO: where should we put these?
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2
import vtkmodules.vtkRenderingFreeType

# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

# for looking for vtk objects
#import vtkmodules.all as vtk
#import vtk

from . import utils
from .utils import (
    dbg_print,
    GetNonconflitName,
    WindowsFriendlyDateTime,
    MergeFullDict,
    Struct,
)
utils.debug_level = 5

from .data_loader import (
    OnDemandVolumeLoader,
)
from .cg_operations import (
    ShotScreen,
    FocusModeController,
)
from .ui_interactions import (
    GenerateKeyBindingDoc,
    PointSetHolder,
    TimerHandler,
)
from .cg_translators import ObjTranslator

# Default configuration parameters are as follow
def DefaultGUIConfig():
    d = {
        "window": {
            "size": "auto",
            "title": "Neuron3DViewer",
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

# Moved from GUIControl.AddBatchSWC() to here,
# so that to solve "Can't pickle local object" problem
def batch_load(file_path_batch, verbosity):
    # somehow, we need this 'verbosity' to pass module level variable
    utils.debug_level = verbosity
    dbg_print(5, '...dealing', file_path_batch)
    results = []
    for f in file_path_batch:
        results.append(ObjTranslator.obj_swc.LoadRawSwc(f))
    return results

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
        self.color_lut = {}
        
        self.utility_objects = {}
        self.volume_loader = OnDemandVolumeLoader()
        
        self.win_conf = {}
        self.scene_saved = {
            'object_properties': {},
            'objects': {}
        }
        self.point_set_holder = PointSetHolder()
        # If a new volume is loaded, it will be clipped by the observers
        self.volume_observers = []
        self.selected_pid = None
        self.focusController = FocusModeController()
        self.timer_handler = TimerHandler()
        self._timer_lazy_render = Struct(finished = True)

        # load default settings
        self.loading_default_config = True
        self.GUISetup(DefaultGUIConfig())
        self.AppendToScene(DefaultSceneConfig())
        self.loading_default_config = False
        self.n_max_cpu_cores_default = 8
        self.swc_loading_batch_size = 2
        self.parallel_lib = 'multiprocessing'

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

    def Set3DCursor(self, xyz):
        # operate on object: 3d_cursor
        if '3d_cursor' not in self.scene_objects:
            # create one
            obj_conf = \
            {
                "type": "Sphere",
            }
            self.AddObject('3d_cursor', obj_conf)
        dbg_print(4, 'Set 3D cursor to', xyz)
        cursor = self.scene_objects['3d_cursor']
        cursor.position = xyz
        self.LazyRender()

    def SetSelectedPID(self, pid):
        if pid < len(self.point_set_holder):
            dbg_print(4, 'SetSelectedPID(): pid =', pid)
            self.selected_pid = pid
            self.focusController.SetCenterPoint(pid)
            self.Set3DCursor(self.point_set_holder()[:,pid])
        else:
            dbg_print(3, 'SetSelectedPID(): pid=%d out of range, len=%d' \
                      %(pid, len(self.point_set_holder)))

    def GetObjectsByType(self, str_type, n_find = -1, visible_only = False):
        """ Find all (at most n_find) objects with the type str_type. """
        li_obj = {}
        for k, conf in self.scene_saved['objects'].items():
            if (str_type is not None) and (conf['type'] != str_type):
                continue
            obj = self.scene_objects[k]
            if visible_only and getattr(obj, 'visible', True) == False:
                continue
            li_obj[k] = obj
            n_find -= 1
            if n_find == 0:
                break
        return li_obj

    def GetDefaultSWCLUT(self):
        if 'default_swc' not in self.color_lut:
            lut = self.translator.prop_lut().parse()
            self.color_lut['default_swc'] = lut

        return self.color_lut['default_swc']

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
            if 'copy_from' in prop_conf:
                # copy reference object property configuration and update it
                dbg_print(4, 'Copy property from "' + prop_conf['copy_from'] + '"')
                name_ref = prop_conf['copy_from']
                prop_conf_ref = self.scene_saved['object_properties'][name_ref]
                prop_conf_z = copy.deepcopy(prop_conf_ref)
                MergeFullDict(prop_conf_z, prop_conf)
                del prop_conf_z['copy_from']
            else:
                prop_conf_z = prop_conf
            object_property = self.translator \
                            .translate_prop_conf(self, prop_conf_z)
            if 'copy_from' in prop_conf:
                # TODO: we might not need these
                object_property.prop_conf = prop_conf_ref
                prop_ref = self.object_properties[name_ref]
                object_property.ref_prop = prop_ref
        else:
            dbg_print(2, 'AddObjectProperty(): unknown property type')

        #if not self.loading_default_config:
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

        dbg_lv = 3
        if name.startswith('_'):
            dbg_lv = 5
        dbg_print(dbg_lv, 'AddObject: "' + name + '" :', obj_conf)
        dbg_print(     4, 'renderer: ',  obj_conf.get('renderer', '0'))

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
        n_job_cores = min(self.n_max_cpu_cores_default, cpu_count())
        dbg_print(4, f'AddBatchSWC(): using {n_job_cores} cores')

        dbg_print(4, 'AddBatchSWC(): loading...')
        # load swc data in parallel
        t1 = time.time()
        if self.parallel_lib == 'joblib':
            # TODO: the parallel execution has a large fixed overhead,
            #       this overhead is not sensitive to batch size,
            #       which indicate the overhead is probably related to
            #       large data transfer (RAM IO?).
            #       The profiler (cProfile) sees
            #       {method 'acquire' of '_thread.lock' objects} which consumes
            #       most of the run time.
            # PS: joblib also has a batch_size, we might use that as well.
            cached_pointsets = joblib.Parallel(n_jobs = n_job_cores) \
                    (joblib.delayed(batch_load)(j, utils.debug_level)
                        for j in li_file_path_batch)
        elif self.parallel_lib == 'multiprocessing':
            with Pool(n_job_cores) as p:
                li_file_path_batch_ext = zip(li_file_path_batch,
                    (utils.debug_level for i in range(len(li_file_path_batch))))
                cached_pointsets = p.starmap(batch_load, li_file_path_batch_ext)
        else:
            # non-parallel version
            cached_pointsets = [batch_load(j, utils.debug_level)
                                for j in li_file_path_batch]
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
        dbg_lv = 3
        if name.startswith('_'):
            dbg_lv = 5
        dbg_print(dbg_lv, 'Removing object:', name)
        # TODO: Do not remove if it is an active camera
        self.scene_objects[name].remove()
        
        if name in self.selected_objects:
            self.selected_objects.remove(name)
        if name in self.point_set_holder.name_idx_map:
            self.point_set_holder.RemovePointsByName(name)
        del self.scene_objects[name]
        if name in self.scene_saved['objects']:
            del self.scene_saved['objects'][name]

    def RemoveBatchObj(self, names):
        self.point_set_holder.RemovePointsByName(names)
        # .copy to avoid on-the-fly for-loop modification of .selected_objects
        for obj_name in names.copy():   
            self.RemoveObject(obj_name)

    def RemoveSelectedObjs(self):
        self.RemoveBatchObj(self.selected_objects)

    def BindKeyToFunction(self, keystoke, uiact_func):
        kbind = self.interactor.style.bind_key_to_function( \
            keystoke, uiact_func)

    def LoadVolumeNear(self, pos, radius=20):
        if (pos is None) or (len(pos) != 3):
            return []
        vol_list = self.volume_loader.LoadVolumeAt(pos, radius)
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
        # get object names of focus_vols
        focus_vols_name_set = set()
        for v in focus_vols:
            focus_vols_name_set.add('volume' + get_vol_name(v['image_path']))
        # collect desired volumes that is not loaded into the scene
        add_set = []
        for vol in focus_vols:
            name = 'volume' + get_vol_name(vol['image_path'])
            if name not in scene_vols:
                add_set.append(vol)
        # scan and remove volumes that is not "focused" and not selected.
        for sv in scene_vols:
            if sv not in focus_vols_name_set and type(scene_vols[sv]) == self.translator.obj_volume:
                if len(self.selected_objects) > 0 and (sv is not self.selected_objects[0]):
                    self.RemoveObject(sv)
        # load the new volumes
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

    def UpdataDynamicVolume(self, origin, block_sz, level = 0):
        pass

    def ShowOnScreenHelp(self):
        if "_help_msg" not in self.scene_objects:
            # show help
            conf = {
                "type": "TextBox",
                "text": GenerateKeyBindingDoc( \
                            self.interactor.style.key_bindings),
                "font_size": "auto",                   # auto or a number
                "font_family": "mono",                 # optional
                "color": [1.0, 1.0, 0.1],
                "background_color": [0.0, 0.2, 0.5],
                "background_opacity": 0.8,             # optional
                "position": "center",                  # optional
                "frame_on": True,                      # optional
            }
            self.AddObject("_help_msg", conf)
        else:
            # remove help
            self.RemoveObject("_help_msg")
        self.render_window.Render()

    def StatusBar(self, msg, timeout = None):
        """Message shown on the button."""
        if (msg is None) and ("_status_bar" in self.scene_objects):
            self.RemoveObject("_status_bar")
            self.LazyRender()
            return
        if "_status_bar" not in self.scene_objects:
            # show help
            conf = {
                "type": "TextBox",
                "text": msg,
                "font_size": "auto",                   # auto or a number
                "font_family": "mono",                 # optional
                "color": [1.0, 1.0, 0.1],
                "background_color": [0.0, 0.2, 0.5],
                "background_opacity": 0.8,             # optional
            }
            self.AddObject("_status_bar", conf)
        else:
            # update message
            self.scene_objects['_status_bar'].text = msg
        # TODO: up to the caller (usually interactor) to update, or lazy update?
        self.LazyRender()

    def Render(self):
        self.GetMainRenderer().Modified()
        self.render_window.Render()

    def LazyRender(self, frame_duration = 1/30):
        self.render_window.Modified()

        if self.timer_handler.interactor is None:
            # not ready (initialized)
            return

        if not self._timer_lazy_render.finished:
            # just wait the scheduled render event
            #dbg_print(5, 'LazyRender(): rejected a Render():')
            return

        def render_if_not_yet(o):
            m_time_cast = self.render_window.GetMTime()
            m_time_ren  = self.GetMainRenderer().GetMTime()
            #dbg_print(5, 'LazyRender(): t_ren =', m_time_ren, ', t_cast =', m_time_cast)
            if m_time_ren <= m_time_cast:
                # no rendering is done, do it now
                self.render_window.Render()
                #dbg_print(5, 'LazyRender(): Render()')

        self._timer_lazy_render = \
            self.timer_handler.schedule(
                render_if_not_yet,
                frame_duration)

    def InfoBar(self, msg):
        # prepare obj message
        if isinstance(msg, dict):
            d_msg = msg
            msg = ''
            if d_msg['type'] == 'swc':
                name = d_msg['obj_name']
                obj = self.scene_objects[name]
                if 'header' in d_msg:
                    msg = d_msg['header'] + '\n'
                else:
                    msg = ''
                msg += f'file name: {obj.swc_name} \n'
                msg += f'index: {name} '
            else:
                dbg_print(1, 'What type:', msg['type'])

        if "_info_bar" not in self.scene_objects:
            # show help
            conf = {
                "type": "TextBox",
                "text": msg,
                "font_size": "auto",                   # auto or a number
                "font_family": "mono",                 # optional
                "color": [0.5, 1.0, 0.1],
                "background_color": [0.0, 0.2, 0.3],
                "background_opacity": 0.8,             # optional
                "position": "lowerright"
            }
            self.AddObject("_info_bar", conf)
        else:
            self.scene_objects["_info_bar"].text = msg
        self.LazyRender()

    def ShowWelcomeMessage(self, show = True):
        dbg_print(4, 'ShowWelcomeMessage(): show =', show)
        welcome_msg = " Drag-and-drop files or a directory here to load data. Press 'h' key to get help."
        if ("_welcome_msg" not in self.scene_objects) and show:
            # show help
            conf = {
                "type": "TextBox",
                "text": welcome_msg,
                "font_size": "auto",                   # auto or a number
                "font_family": "mono",                 # optional
                "color": [1.0, 1.0, 0.1],
                "background_color": [0.0, 0.0, 0.0],
                "background_opacity": 0.0,             # optional
            }
            self.loading_default_config = True
            self.AddObject("_welcome_msg", conf)
            self.loading_default_config = False
        elif ("_welcome_msg" in self.scene_objects) and not show:
            self.RemoveObject("_welcome_msg")
            self.render_window.Render()

    def EasyObjectImporter(self, cmd_obj_desc):
        """
        Used to accept command line inputs which need default parameters.
        """
        if not cmd_obj_desc:
            return
        if isinstance(cmd_obj_desc, str):
            self.DropFilesObjectImporter([cmd_obj_desc])
            return
        if isinstance(cmd_obj_desc, list):
            self.DropFilesObjectImporter(cmd_obj_desc)
            return

        self.li_cg_conf = self.translator \
                          .parse_all_cmd_args_obj(cmd_obj_desc, 'animation')

        tl_win = self.translator.init_window(self, None)
        win_conf = tl_win.parse_cmd_args(cmd_obj_desc)
        tl_win.parse(win_conf)
        tl_win.parse_post_renderers(win_conf)
        
        # so far, init_renderers, init_interactor do not have cmd options.
        # except
        if 'plugin_dir' in cmd_obj_desc:
            self.interactor.style.refresh_key_bindings()
        
        scene_ext = self.translator.init_scene \
                    .parse_cmd_args(cmd_obj_desc)
        self.AppendToScene(scene_ext)

        li_obj_conf = self.translator.parse_all_cmd_args_obj(cmd_obj_desc)
        
        # for load SWC in parallel
        li_swc_conf = [o for o in li_obj_conf if o['type'] == 'swc']
        li_obj_conf = [o for o in li_obj_conf if o['type'] != 'swc']

        if len(li_swc_conf) > 0:
            self.AddBatchSWC('swc', li_swc_conf)
        
        for obj_conf in li_obj_conf:
            name = self.GetNonconflitName(obj_conf['type'])
            self.AddObject(name, obj_conf)

    def DropFilesObjectImporter(self, file_path_list):
        self.ShowWelcomeMessage(False)
        # put each file to each category
        swc_file_list = []
        swc_dir_list = []
        img_file_list = []
        for file_path in file_path_list:
            if file_path.endswith('.swc'):
                swc_file_list.append(file_path)
            elif file_path.endswith('.tif') \
              or file_path.endswith('.tiff') \
              or file_path.endswith('.ims') \
              or file_path.endswith('.h5'):
                img_file_list.append(file_path)
            elif os.path.isdir(file_path):
                swc_dir_list.append(file_path)
            else:
                dbg_print(2, 'Unrecognized object:', file_path)
        
        # swc type
        if len(swc_file_list) > 0:
            self.EasyObjectImporter({'swc': swc_file_list})
        # swc_dir type
        for swc_dir in swc_dir_list:
            self.EasyObjectImporter({'swc_dir': swc_dir})
        # img file
        for img_path in img_file_list:
            self.EasyObjectImporter({'img_path': img_path})
        # see it
        self.GetMainRenderer().ResetCamera()
        self.LazyRender()

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

    def Start(self, run_fn = None):
        self.interactor.Initialize()  # must be called prior to creating timer
        self.GetMainRenderer().ResetCamera()   # bird's-eye view
        #self.GetMainRenderer().ResetCameraClippingRange()
        self.render_window.Render()
        # self.UtilizerInit()
        self.focusController.SetGUIController(self)
        self.timer_handler.Initialize(self.interactor)

        if run_fn:
            if not isinstance(run_fn, list):
                run_fn = [run_fn]
            for fn in run_fn:
                fn(self)

        for cg_conf in self.li_cg_conf:
            self.translator.translate(self, self.GetMainRenderer(),
                                      'animation_', cg_conf)
        
        if not self.do_not_start_interaction:
            self.interactor.Start()

def get_program_parameters():
    description = 'Portable volumetric image and neuronal tracing result viewer based on PyVTK.'
    author = 'Author: https://github.com/bewantbe/SimpleVolumeViewer'
    epilogue = GenerateKeyBindingDoc()
    parser = argparse.ArgumentParser( 
                argument_default=argparse.SUPPRESS,  # ignore non-specified options
                description=description,
                epilog=epilogue + author + '\n \n',
                formatter_class=argparse.RawDescriptionHelpFormatter)
    ObjTranslator().add_all_arguments_to(parser)
    parser.add_argument(
        '--verbosity', type=int, metavar='INT',
        help="""
        Debug message verbosity:
          0 = show nothing unless crash, 
          1 = error, 
          2 = warning, 
          3 = hint, 
          4 = message, 
          5 >= verbose.
          """)
    args = parser.parse_args()
    if getattr(args, 'verbosity', None):
        utils.debug_level = args.verbosity
    # convert args from Namespace to dict, and filter out None values.
    #args = {k:v for k,v in vars(args).items() if v is not None}
    args = vars(args)
    dbg_print(3, 'get_program_parameters(): d=', args)
    return args

def main():
    #freeze_support()
    cmd_obj_desc = get_program_parameters()
    gui = GUIControl()
    t1 = time.time()
    dbg_print(4, f'Init time: {t1 - g_t0:.3f} sec.')
    if len(cmd_obj_desc.keys()) == 0:
        gui.ShowWelcomeMessage()
    gui.EasyObjectImporter(cmd_obj_desc)
    gui.Start()
    te = time.time()
    dbg_print(4, f'Wall time: {te - g_t0:.3f} sec.')
