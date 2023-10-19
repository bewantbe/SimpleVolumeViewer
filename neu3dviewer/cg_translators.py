# SPDX-License-Identifier: GPL-3.0-or-later

# Computer graphics translators, the real workhorse.
# You may also call them operators (of VTK), but "translator" is descriptive
# style, "operator" is imperative style, I prefer the former, so everything
# here is passive and the control point is left in the main code (GUIControl).

# Ref. for VTK
# Python Wrappers for VTK
# https://vtk.org/doc/nightly/html/md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html

# Demonstrates physically based rendering using image based lighting and a skybox.
# https://kitware.github.io/vtk-examples/site/Python/Rendering/PBR_Skybox/

import os            # for os.path
import numpy as np
from numpy import array as _a
from numpy.random import randint
import json
import numbers

from vtkmodules.vtkCommonCore import (
    vtkCommand,
    vtkPoints,
    VTK_CUBIC_INTERPOLATION,
    vtkLookupTable,
)
from vtkmodules.vtkCommonDataModel import (
    vtkPiecewiseFunction,
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine,
)
from vtkmodules.vtkIOGeometry import (
    vtkOBJReader,
    vtkSTLReader,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
    vtkColorSeries,
)
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkColorTransferFunction,
    vtkCamera,
    vtkVolume,
    vtkVolumeProperty,
    vtkActor,
    vtkPolyDataMapper,
    vtkTextActor,
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.util.numpy_support import numpy_to_vtk

from .utils import (
    vtkGetColorAny3d,
    vtkGetColorAny4d,
    _mat3d,
    dbg_print,
    UpdatePropertyOTFScale,
    UpdatePropertyCTFScale,
    GetColorScale,
    SetColorScale,
    ConditionalAddItem,
    WindowsFriendlyDateTime,
)
from .data_loader import *
from .ui_interactions import (
    MyInteractorStyle,
    execSmoothRotation,
)

def AlignCameraDirection(cam2, cam1, dist=4.0):
    """
    Align direction of cam2 by cam1, and make cam2 dist away from origin.
    """
    r = np.array(cam1.GetPosition()) - np.array(cam1.GetFocalPoint())
    r = r / np.linalg.norm(r) * dist

    cam2.SetRoll(cam1.GetRoll())
    cam2.SetPosition(r)
    cam2.SetFocalPoint(0, 0, 0)
    cam2.SetViewUp(cam1.GetViewUp())

def CameraFollowCallbackFunction(caller, ev):
    cam1 = CameraFollowCallbackFunction.cam1
    cam2 = CameraFollowCallbackFunction.cam2
    AlignCameraDirection(cam2, cam1)
    return

class ObjTranslator:
    """
    The collection of translators to convert json description to computer 
      graphic objects.
    The ideal is that eventually GUIControl do not contain any
      implementation details.
    All the translator units should not have state.
    Also handle command line parse.
    External code should be easy to modify this class to extend its function.
    """

    class TranslatorUnit:
        """
        Represent a graphic object, at its core is VTK actor.
        For common manipulations (add/remove/move position etc.), use
          member functions in this class instead of the underlying VTK actor.

        Call procedure:
            (1) add_argument_to()
                    Called by command line parser before the parsing.
            (2) parse_cmd_args():
                    Called by GUIControl.EasyObjectImporter to convert the 
                    command line options to json-style scene/object 
                    description.
            (3) parse():
                    Translate the json-style scene/object description to
                    concrete VTK object(s) or scene(window, renderer etc.) 
                    settings.
        """
        #@staticmethod
        #def add_argument_to(args_parser):
        #    pass

        #@staticmethod
        #def parse_cmd_args(cmd_obj_desc):
        #    pass

        def __init__(self, gui_ctrl, renderer):
            self.gui_ctrl = gui_ctrl
            self.renderer = renderer
            self.actor = None          # the VTK actor object
            self.position = None       # can be used to move the object

        def parse(self, st):
            """
            translate json description to obj on screen.
            """
            pass

        def remove(self, rm_conf = {}):
            """
            Default remove function.
            """
            if not rm_conf:
                # remove the whole object
                if self.actor:
                    self.renderer.RemoveActor(self.actor)
                    self.actor = None

        @property
        def position(self):
            return self._position
        
        @position.setter
        def position(self, ext_pos):
            if (self.actor is not None) and (ext_pos is not None):
                self.actor.SetPosition(ext_pos)
            self._position = ext_pos

    # dispatch functions
    def translate(self, gui_ctrl, renderer, prefix_class, obj_conf):
        """
        prefix_class = 'obj_' or 'prop_'
        """
        obj_type = obj_conf['type']
        tl_unit = getattr(self, prefix_class + obj_type)(gui_ctrl, renderer)
        return tl_unit.parse(obj_conf)

    def translate_obj_conf(self, gui_ctrl, renderer, obj_conf):
        return self.translate(gui_ctrl, renderer, 'obj_', obj_conf)

    def translate_prop_conf(self, gui_ctrl, prop_conf):
        return self.translate(gui_ctrl, None, 'prop_', prop_conf)

    def get_all_tl_unit(self, tl_prefix = []):
        if isinstance(tl_prefix, str):
            tl_prefix = [tl_prefix]
        return [getattr(self, a)
                  for a in dir(self)
                    if (not a.startswith('_')) and 
                       (a != 'TranslatorUnit') and
                       isinstance(getattr(self, a), type) and
                       issubclass(getattr(self, a),
                                  ObjTranslator.TranslatorUnit) and
                       (not tl_prefix or (a.split('_',1)[0] in tl_prefix))]

    # batch add arguments to parser (argparse)
    def add_all_arguments_to(self, args_parser):
        for tl_u in self.get_all_tl_unit():
            if hasattr(tl_u, 'add_argument_to'):
                getattr(tl_u, 'add_argument_to')(args_parser)

    def parse_all_cmd_args_obj(self, cmd_obj_desc, 
                                     tl_prefix = ['obj', 'prop']):
        li_obj_conf = []
        for tl_u in self.get_all_tl_unit(tl_prefix):
            if hasattr(tl_u, 'parse_cmd_args'):
                c = getattr(tl_u, 'parse_cmd_args')(cmd_obj_desc)
                if isinstance(c, list):
                    li_obj_conf.extend(c)
                elif c is not None:
                    li_obj_conf.append(c)
        return li_obj_conf

    # Translation units for initialization of GUI

    class init_window(TranslatorUnit):
        """
        prototype:
        "window": {
            "size": [2400, 1800],
            "title": "SimpleRayCast",
            "number_of_layers": 2,
            "stereo_type": "SplitViewportHorizontal",
            "max_cpu": 8,
            "swc_loading_batch_size": 2,
            "full_screen": 1,
            "off_screen_rendering": 0,
            "no_interaction": 0,
            "plugin_dir": "./plugins/"
        },
        """

        @staticmethod
        def add_argument_to(parser):
            group = parser.add_argument_group('Window settings')
            group.add_argument('--window_size', metavar='SIZE',
                    help='Set window size like "1024x768".')
            group.add_argument('--off_screen_rendering',
                    type=int, choices=[0, 1],
                    help='Enable off-screen rendering. 1 = Enable, 0 = Disable.')
            group.add_argument('--no_interaction', type=int, choices=[0, 1],
                    help='Used with --screenshot, for exit after the screenshot(1).')
            group.add_argument('--full_screen', type=int, choices=[0, 1],
                    help='Full screen On(1)/Off(0).')
            group.add_argument('--stereo_type',
                    help=
                    """
                    Enable stereo rendering, set the type here.
                    Possible types:
                    CrystalEyes, RedBlue, Interlaced, Left, Right, Fake, Emulate,
                    Dresden, Anaglyph, Checkerboard, SplitViewportHorizontal
                    """)
            group.add_argument('--max_cpu', type=int, metavar='N_CPU',
                    help="""Max number of CPUs for parallel computing,
                            default: 8 or max number of CPUs in your system.
                            Currently only for loading SWC.""")
            group.add_argument('--swc_loading_batch_size', type=int, metavar='SIZE',
                    help='The batch size for each CPU when loading SWC files, default: 2.')
            group.add_argument('--parallel_lib', type=str, metavar='LIB',
                    choices=['multiprocessing', 'joblib', 'none'],
                    help='Which multi-process parallel library to use.')
            group.add_argument('--plugin_dir',
                    help=f'Set directory of plugins. Default: ./plugins/')

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            win_conf = {}
            li_name = {
                'off_screen_rendering':'', 'no_interaction':'',
                'full_screen':'', 'stereo_type':'',
                'max_cpu':'', 'swc_loading_batch_size':'',
                'parallel_lib':'',
                'plugin_dir':'',
                'window_size':'size',
            }
            for name, key_name in li_name.items():
                ConditionalAddItem(name, cmd_obj_desc, key_name, win_conf)
            return win_conf

        def parse(self, win_conf):
            # TODO: return render_window object to gui_ctrl
            # TODO: try vtkVRRenderWindow?
            if self.gui_ctrl.render_window is None:
                self.gui_ctrl.render_window = vtkRenderWindow()
                self.gui_ctrl.render_window.StereoCapableWindowOn()
            render_window = self.gui_ctrl.render_window
            if 'size' in win_conf:
                if win_conf['size'] == 'auto':
                    # note: on VTK 9.0 & 9.1 calling .GetScreenSize() can
                    # cause segmentation fault
                    # https://discourse.vtk.org/t/vtk9-1-0-problems/7094/5
                    # Crash on window close with vtkXRenderWindowInteractor
                    # https://gitlab.kitware.com/vtk/vtk/-/issues/18372
                    screen_size = render_window.GetScreenSize()
                    self.screen_size = screen_size
                    # Use 60% of the vertical space, with 4:3 aspect ratio.
                    # But if the resolution is too low (win_size[y] < 800),
                    #   use 800 pixels high or up to the screen size[y]
                    y = 0.8 * screen_size[1]
                    wnd_y_min = 800
                    aspect_ratio = 4/3
                    if y < wnd_y_min:
                        y = min(wnd_y_min, screen_size[1])
                    win_size = [aspect_ratio * y, y]
                else:
                    win_size = str2array(win_conf['size'], 'x', int)
                win_size = list(map(int, win_size))
                render_window.SetSize(win_size)
            if 'title' in win_conf:
                render_window.SetWindowName(win_conf['title'])
            if 'number_of_layers' in win_conf:
                render_window.SetNumberOfLayers(
                    win_conf['number_of_layers'])
            if ('stereo_type' in win_conf) and win_conf['stereo_type']:
                t = win_conf['stereo_type']
                if t == 'CrystalEyes':
                    render_window.SetStereoTypeToCrystalEyes()
                elif t == 'RedBlue':
                    render_window.SetStereoTypeToRedBlue()
                elif t == 'Interlaced':
                    render_window.SetStereoTypeToInterlaced()
                elif t == 'Left':
                    render_window.SetStereoTypeToLeft()
                elif t == 'Right':
                    render_window.SetStereoTypeToRight()
                elif t == 'Dresden':
                    render_window.SetStereoTypeToDresden()
                elif t == 'Anaglyph':
                    render_window.SetStereoTypeToAnaglyph()
                elif t == 'Checkerboard':
                    render_window.SetStereoTypeToCheckerboard()
                elif t == 'SplitViewportHorizontal':
                    render_window.SetStereoTypeToSplitViewportHorizontal()
                elif t == 'Fake':
                    render_window.SetStereoTypeToFake()
                elif t == 'Emulate':
                    render_window.SetStereoTypeToEmulate()
                render_window.StereoRenderOn()
            else:
                render_window.StereoRenderOff()
                #render_window.StereoCapableWindowOn()
            if 'max_cpu' in win_conf:
                self.gui_ctrl.n_max_cpu_cores_default = win_conf['max_cpu']
            if 'swc_loading_batch_size' in win_conf:
                self.gui_ctrl.swc_loading_batch_size = win_conf['swc_loading_batch_size']
            if 'parallel_lib' in win_conf:
                self.gui_ctrl.parallel_lib = win_conf['parallel_lib']
            self.gui_ctrl.plugin_dir = win_conf.get('plugin_dir', './plugins/')
            # The get DPI function in VTK is very unusable.
            #if render_window.DetectDPI():
            #    render_window.dpi = render_window.GetDPI()
            #    dbg_print(4, 'DPI:', render_window.dpi)
            #    render_window.SetDPI(96)
            #    dbg_print(4, 'set DPI = 267')
            #    render_window.dpi = render_window.GetDPI()
            #    dbg_print(4, 'DPI:', render_window.dpi)
            #else:
            #    render_window.dpi = 96
            #    dbg_print(4, 'DPI default:', render_window.dpi)
            if 'full_screen' in win_conf:
                window_size_old   = render_window.GetSize()
                window_position_old = render_window.GetPosition()
                screen_size = render_window.GetScreenSize()
                render_window.SetFullScreen(win_conf['full_screen']>0)
                dbg_print(5, 'screen size:', screen_size)
                window_size_full = render_window.GetSize()
                dbg_print(5, 'window size:', window_size_full)
                if win_conf['full_screen']:
                    # non full screen -> full screen
                    # note down old window size and position
                    render_window._windows_size = window_size_old
                    render_window._windows_position = window_position_old
                    # Probably a VTK bug:
                    # force window size to be the same as the screen size.
                    if np.any(_a(window_size_full) - _a(screen_size)):
                        render_window.SetSize(screen_size)
                        dbg_print(5, 'Forced window to size:', render_window.GetSize())
                else:
                    if hasattr(render_window, '_windows_position'):
                        dbg_print(4, 'Restoring window position and size.')
                        render_window.SetPosition(render_window._windows_position)
                        render_window.SetSize(render_window._windows_size)
                    else:
                        render_window.InvokeEvent(vtkCommand.WindowResizeEvent)
            # note down
            self.gui_ctrl.win_conf.update(win_conf)

        def parse_post_renderers(self, win_conf):
            # Off screen rendering
            # https://discourse.vtk.org/t/status-of-vtk-9-0-with-respect-to-off-screen-rendering-under-ubuntu-with-pip-install/5631/2
            if win_conf.get('off_screen_rendering', False) > 0:
                self.gui_ctrl.render_window.SetOffScreenRendering(1)
                self.gui_ctrl.do_not_start_interaction = True
            else:
                self.gui_ctrl.do_not_start_interaction = False
            # Hint: you may use xdotoll to control an off-screen program
            # e.g. xdotool search --name SimpleRayCast key 'q'
            if 'no_interaction' in win_conf:
                self.gui_ctrl.do_not_start_interaction = win_conf['no_interaction'] > 0
            # note down
            self.gui_ctrl.win_conf.update(win_conf)

    class init_renderers(TranslatorUnit):
        """
        prototype:
        "renderers":{
            "0":{
                "layer": 0
            },
            "1":{
                "layer": 1,
                "view_port": [0.0, 0.0, 0.2, 0.2]
            }
        }
        """
        def parse(self, renderers_conf):
            # Ref: Demonstrates the use of two renderers. Notice that the second (and subsequent) renderers will have a transparent background.
            # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
            # get our renderer list
            renderers = self.renderer
            render_window = self.gui_ctrl.render_window
            # load new renderers
            for key, ren_conf in renderers_conf.items():
                if key in renderers:
                    # remove old renderer
                    render_window.RemoveRenderer(renderers[key])
                # https://kitware.github.io/vtk-examples/site/Python/Rendering/TransparentBackground/
                # setup new renderer
                renderer = vtkRenderer()
                if 'layer' in ren_conf:
                    renderer.SetLayer(ren_conf['layer'])
                if 'view_port' in ren_conf:
                    renderer.SetViewport(ren_conf['view_port'])
                #renderer.UseFXAAOn()  # fast
                renderers[key] = renderer
                # add new renderer to window
                render_window.AddRenderer(renderer)
                #render_window.SetMultiSamples(8)  # high quality, default?

    class init_interactor(TranslatorUnit):
        def parse(self, ui_conf):
            if (not ui_conf) and (self.gui_ctrl.interactor):
                # it is initialized, so do not change anything
                return self.gui_ctrl.interactor
            # Create the interactor (for keyboard and mouse)
            interactor = vtkRenderWindowInteractor()
            style = MyInteractorStyle(interactor, self.gui_ctrl)
            interactor.SetInteractorStyle(style)
        #    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
            interactor.SetRenderWindow(self.gui_ctrl.render_window)
            #interactor.SetDesiredUpdateRate(30.0)
            interactor.style = style
            return interactor

    class init_scene(TranslatorUnit):
        """
        {
            "object_properties": {...},
            "objects": {}
        }
        """

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--scene', metavar='FILE_PATH',
                    help='Project scene file path. e.g. for batch object loading.')

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            if 'scene' in cmd_obj_desc:
                # TODO: maybe move this before init of gui, and pass it as init param.
                scene_ext = json.loads(open(cmd_obj_desc['scene']).read())
            else:
                scene_ext = {}
            return scene_ext

        def parse(self, obj_conf):
            pass

    class prop_volume(TranslatorUnit):
        """
        prototype:
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
        """
        def parse(self, prop_conf):
            volume_property = vtkVolumeProperty()
            
            if 'opacity_transfer_function' in prop_conf:
                otf_conf = prop_conf['opacity_transfer_function']
                otf_v = otf_conf['AddPoint']
                otf_s = otf_conf.get('opacity_scale', 1.0)
                # perform scaling
                otf_v_e = np.array(otf_v).copy()
                for v in otf_v_e:
                    v[0] = v[0] *  otf_s
                # Create transfer mapping scalar value to opacity.
                otf = vtkPiecewiseFunction()
                for v in otf_v_e:
                    otf.AddPoint(*v)
                volume_property.SetScalarOpacity(otf)

            if 'color_transfer_function' in prop_conf:
                ctf_conf = prop_conf['color_transfer_function']
                ctf_v = ctf_conf['AddRGBPoint']
                ctf_s = ctf_conf.get('trans_scale', 1.0)
                # perform scaling
                ctf_v_e = np.array(ctf_v).copy()
                for v in ctf_v_e:
                    v[0] = v[0] *  ctf_s
                # Create transfer mapping scalar value to color.
                ctf = vtkColorTransferFunction()
                for v in ctf_v_e:
                    ctf.AddRGBPoint(*v)
                volume_property.SetColor(ctf)

            # shading only valid for blend mode vtkVolumeMapper::COMPOSITE_BLEND
            volume_property.ShadeOn()

            if 'interpolation' in prop_conf:
                if prop_conf['interpolation'] == 'cubic':
                    volume_property.SetInterpolationType(
                        VTK_CUBIC_INTERPOLATION)
                elif prop_conf['interpolation'] == 'linear':
                    volume_property.SetInterpolationTypeToLinear()
                else:
                    dbg_print(2, 'AddObjectProperty(): unknown interpolation type')
            volume_property.prop_conf = prop_conf
            return volume_property
        
        def modify(self, obj_prop, prop_conf):
            dbg_print(4, 'ModifyObjectProperty():')
            #if name.startswith('volume'):
            # both obj_prop and prop_conf will be updated
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

    class prop_lut(TranslatorUnit):
        """
        {
            "type": "lut",
            "lut": "default"
        }
        """
        def __init__(self):
            pass

        def parse(self, prop_conf = None):
            if prop_conf is None:
                e_lut = 'default'
            else:
                e_lut = prop_conf['lut']

            lut = vtkLookupTable()
            if isinstance(e_lut, str) and (e_lut == 'default'):
                # default color lookup table (colormap)
                # the number 402 (~256*pi/2) as suggested by lut.SetRampToSCurve()
                lut.SetNumberOfTableValues(402)
                lut.Build()
            elif isinstance(e_lut, (list, np.ndarray)):
                # assume e_lut are colors:
                #   [(r,g,b), ...]
                #   [(r,g,b,a), ...]
                #  r,g,b,a are in 0~1
                n_color = len(e_lut)
                dbg_print(4, 'prop_lut::custom lut n =', n_color)
                if len(e_lut[0]) == 3:
                    # convert to RGBA
                    e_lut = np.hstack((e_lut, np.ones((n_color,1))))
                lut.SetNumberOfTableValues(n_color)
                for j, c in enumerate(e_lut):
                    lut.SetTableValue(j, e_lut[j])
                lut.SetNanColor(0.5, 0.0, 0.0, 1.0)
                lut.SetBelowRangeColor(e_lut[0])
                lut.SetAboveRangeColor(e_lut[-1])
                lut.SetTableRange(0.0, 1.0)
                lut.SetRampToLinear()
                lut.SetScaleToLinear()
            return lut

    class obj_volume(TranslatorUnit):
        """
        prototype:
        "volume": {
            "type": "volume",
            "property": "volume"
            "mapper": "GPUVolumeRayCastMapper",
            "mapper_blend_mode": "MAXIMUM_INTENSITY_BLEND",
            "view_point": "auto",
            "file_path": file_path,
            "origin": [100, 200, 300],
            "rotation_matrix": [1,0,0, 0,1,0, 0,0,1],
            "oblique_image": False,
            "colorscale": 4.0,
            # IMS specific
            "level": "0",
            "channel": "0",
            "time_point": "0",
            "range": "[10:100,10:100,10:100]"
            # zarr specific
            "range": "[10:100,10:100,10:100]"
        }
        """

        @staticmethod
        def add_argument_to(parser):
            group = parser.add_argument_group('Volume(3D) image related options')
            group.add_argument('--img_path', metavar='IMG_PATH',
                    help='Image stack file path, can be TIFF or IMS(HDF5).')
            group.add_argument('--level',
                    help='For multi-level image (.ims), load only that level.')
            group.add_argument('--channel',
                    help='Select channel for IMS image.')
            group.add_argument('--time_point',
                    help='Select time point for IMS image.')
            group.add_argument('--range',
                    help='Select range within image, e.g. "[0:100,:,:]".')
            group.add_argument('--colorscale',
                    help='Set scale of color transfer function, a positive number.')
            group.add_argument('--origin', metavar='ORIGIN_COOR',
                    help='Set coordinate of the origin of the volume.')
            group.add_argument('--rotation_matrix',
                    help='Set rotation matrix of the volume, a 9 number list.')
            group.add_argument('--oblique_image', metavar='BOOL',
                    help='Specify the image is imaged oblique, instead of guessing.')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if 'img_path' not in obj_desc:
                return None

            file_path = obj_desc['img_path']
            obj_conf = {
                "type": "volume",
                "view_point": "auto",
                "file_path": file_path
            }
            if file_path.endswith('.tif'):
                # a tiff file
                pass
            elif file_path.endswith('.ims') or file_path.endswith('.h5'):
                # a IMS/HDF5 volume
                obj_conf.update({
                    "level": obj_desc.get('level', '0'),
                    "channel": obj_desc.get('channel', '0'),
                    "time_point": obj_desc.get('time_point', '0'),
                    "range": obj_desc.get('range', '[:,:,:]')
                })
            elif os.path.isdir(file_path) and os.path.isfile(file_path + '/.zarray'):
                # a zarr volume
                obj_conf.update({
                    "range": obj_desc.get('range', '[:,:,:]')
                })
            else:
                dbg_print(1, 'Unrecognized image format.')
                return None
            
            if 'origin' in obj_desc:
                obj_conf.update({
                    'origin': str2array(obj_desc['origin'])
                })
            if 'rotation_matrix' in obj_desc:
                obj_conf.update({
                    'rotation_matrix': str2array(obj_desc['rotation_matrix'])
                })
            if 'oblique_image' in obj_desc:
                obj_conf.update({
                    'oblique_image': obj_desc['oblique_image'].lower() \
                                     in ['true', '1']
                })
            
            if 'colorscale' in obj_desc:
                s = float(obj_desc['colorscale'])
                obj_conf.update({'property': {
                    'type'     : 'volume',
                    'copy_from': 'volume',
                    'opacity_transfer_function': {'opacity_scale': s},
                    'color_transfer_function'  : {'trans_scale': s}
                }})
            else:
                obj_conf.update({'property': 'volume'})

            return obj_conf

        def parse(self, obj_conf):
            file_path = obj_conf['file_path']
            img_importer = ImportImageFile(file_path, obj_conf)
            # TODO: try
            # img_importer = ImportImageArray(file_path.img_array, file_path.img_meta)
            # set position scaling and direction
            img_importer.SetDataOrigin(obj_conf.get('origin', [0,0,0]))
            # for 3d rotation and scaling
            dir3d = img_importer.GetDataDirection()
            idmat = [1,0,0,0,1,0,0,0,1]
            rot3d = obj_conf.get('rotation_matrix', idmat)
            dir3d = (_mat3d(rot3d) @ _mat3d(dir3d)).flatten()
            img_importer.SetDataDirection(dir3d)

            # vtkVolumeMapper
            # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
            mapper_name = obj_conf.get('mapper', 'GPUVolumeRayCastMapper')
            if mapper_name == 'GPUVolumeRayCastMapper':
                volume_mapper = vtkGPUVolumeRayCastMapper()
            elif mapper_name == 'FixedPointVolumeRayCastMapper':
                volume_mapper = vtkFixedPointVolumeRayCastMapper()
            else:
                # TODO: consider use vtkMultiBlockVolumeMapper
                # OR: vtkSmartVolumeMapper https://vtk.org/doc/nightly/html/classvtkSmartVolumeMapper.html#details
                # vtkOpenGLGPUVolumeRayCastMapper
                volume_mapper = vtkGPUVolumeRayCastMapper()
            #volume_mapper.SetBlendModeToComposite()
            volume_mapper.SetInputConnection(img_importer.GetOutputPort())
            # Set blend mode (such as MIP)
            blend_modes = getattr(volume_mapper, 
                obj_conf.get('mapper_blend_mode', 'MAXIMUM_INTENSITY_BLEND'))
            # Possible blend_modes:
            # COMPOSITE_BLEND
            # MAXIMUM_INTENSITY_BLEND
            # MINIMUM_INTENSITY_BLEND
            # AVERAGE_INTENSITY_BLEND
            # ADDITIVE_BLEND
            # ISOSURFACE_BLEND
            # SLICE_BLEND
            # Ref: https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html#aac00c48c3211f5dba0ca98c7a028e409ab4e0747ca0bcf150fa57a5a6e9a34a14
            volume_mapper.SetBlendMode(blend_modes)

            # get property used in rendering
            prop_conf_ref = obj_conf.get('property', 'volume')
            if isinstance(prop_conf_ref, dict):
                # add a new property
                prop_name = self.gui_ctrl.GetNonconflitName('volume', 'property')
                dbg_print(3, 'AddObject(): Adding prop:', prop_name)
                self.gui_ctrl.AddObjectProperty(prop_name, prop_conf_ref)
                volume_property = self.gui_ctrl.object_properties[prop_name]
            else:
                # name of existing property, e.g. 'volume'
                dbg_print(3, 'AddObject(): Using existing prop:', prop_conf_ref)
                volume_property = self.gui_ctrl.object_properties[prop_conf_ref]

            # The volume holds the mapper and the property and
            # can be used to position/orient the volume.
            volume = vtkVolume()
            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            for ob in self.gui_ctrl.volume_observers: ob.Notify(volume)
            self.renderer.AddVolume(volume)

            view_point = obj_conf.get('view_point', 'auto')
            if view_point == 'auto':
                # auto view all actors
                self.renderer.ResetCamera()
            
            self.actor = volume
            return self

        def modify(self, mo_conf, obj_conf):
            """
            {
                "clipping_planes": [(origin_vec3d, normal_vec3d), ...],
                # or
                "clipping_planes": "clipping_planes_group1",
            }
            """
            obj_conf.update(mo_conf)

        def set_color_scale_mul_by(self, k):
            obj_prop = self.actor.GetProperty()
            #obj_prop = self.gui_ctrl.object_properties[vol_name]
            cs_o, cs_c = GetColorScale(obj_prop)
            SetColorScale(obj_prop, [cs_o*k, cs_c*k])

        def get_center(self):
            if not self.actor:
                return None

            bd = self.actor.GetBounds()
            center = [(bd[0]+bd[1])/2, (bd[2]+bd[3])/2, (bd[4]+bd[5])/2]
            return center

    class obj_lychnis_blocks(TranslatorUnit):
        """
        prototype:
        {
            "type"     : "lychnis_blocks",
            "file_path": obj_desc['lychnis_blocks'],
        }
        """

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--lychnis_blocks', metavar='PATH',
                    help='Path of lychnis "blocks.json".')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if 'lychnis_blocks' in obj_desc:
                obj_conf = {
                    "type"     : "lychnis_blocks",
                    "file_path": obj_desc['lychnis_blocks'],
                }
            else:
                obj_conf = None
            return obj_conf

        def parse(self, obj_conf):
            self.gui_ctrl.volume_loader.ImportLychnixVolume( \
                obj_conf['file_path'])
            #TODO: we might return ourself instead of volume_loader
            return self.gui_ctrl.volume_loader

    class obj_swc(TranslatorUnit):
        """
        prototype:
        "swc": {
            "type": "swc",
            "file_path": "RM006-004-lychnis/F5.json.swc",
            "color": "Tomato",
            "opacity": 1.0,        # optional
            "line_width": 2.0,
        }
        """

        @staticmethod
        def add_argument_to(parser):
            group = parser.add_argument_group('SWC (neuron fiber) file options')
            group.add_argument('--swc', action='append', metavar='FILE_PATH',
                    help='Read and draw SWC file. Note: SWC nodes must sorted.')
            group.add_argument('--fibercolor', metavar='COLOR',
                    help='Set fiber color, like "Red".')
            group.add_argument('--line_width',
                    help='Set fiber line width.')
            group.add_argument('--swc_dir',
                    help='Read and draw SWC files in the directory.')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if ('swc' not in obj_desc) and ('swc_dir' not in obj_desc):
                return None

            if 'swc' not in obj_desc:  # for swc_dir
                obj_desc['swc'] = []

            if 'swc_dir' in obj_desc:
                swc_dir = obj_desc['swc_dir']
                if os.path.isdir(swc_dir):
                    # note down *.swc files it to obj_desc['swc']
                    import glob
                    fns = glob.glob(swc_dir + '/*.swc')
                    if len(fns) == 0:
                        dbg_print(2, f'Option --swc_dir "{swc_dir}" contains no swc.')
                    obj_desc['swc'].extend(fns)
                else:
                    dbg_print(1, f'Option --swc_dir "{swc_dir}" not a directory.')
        
            if not isinstance(obj_desc['swc'], (list, tuple)):
                obj_desc['swc'] = [obj_desc['swc'],]
            # See also https://vtk.org/doc/nightly/html/classvtkColorSeries.html#details
            # https://www.kitware.com/off-screen-rendering-through-the-native-platform-interface-egl/
            # possible series:
            #   SPECTRUM (7)
            #   BREWER_DIVERGING_PURPLE_ORANGE_11
            color_scheme = vtkColorSeries()
            color_scheme.SetColorScheme(color_scheme.BREWER_DIVERGING_SPECTRAL_11)
            # color was 'Tomato'
            li_obj_conf = []
            for id_s in range(len(obj_desc['swc'])):
                c = color_scheme.GetColorRepeating(1+id_s)
                c = list(_a([c[0], c[1], c[2]]) / 255.0)
                obj_conf = {
                    "type": "swc",
                    "file_path": obj_desc['swc'][id_s],
                    "color": obj_desc.get('fibercolor', c),
                    "line_width": obj_desc.get('line_width', "auto"),
                }
                li_obj_conf.append(obj_conf)

            return li_obj_conf

        def __init__(self, gui_ctrl, renderer):
            super().__init__(gui_ctrl, renderer)
            # file path of the source SWC data
            self.file_path  = None
            # essentially file name
            self.swc_name   = None
            # for data structure see def LoadSWCTree(filepath)
            self.tree_swc   = None
            # fields for properties
            self._visible   = True
            self._color     = None
            self._opacity   = None
            self._line_width = None
            self._color_lut = None
        
        def parse(self, obj_conf):
            t0 = time.time()
            
            if not hasattr(self, 'cache1'):
                processes, raw_points, ntree = \
                    self.LoadRawSwc(obj_conf['file_path'])
                self.cache1 = (processes, raw_points, ntree)
            else:
                processes, raw_points, ntree = self.cache1
            del self.cache1      # release the variables to speed up

            self.raw_points = raw_points

            self.file_path  = obj_conf['file_path']
            self.swc_name   = os.path.splitext(os.path.basename(self.file_path))[0]
            self.tree_swc   = ntree
            self.processes = processes

            # ref: 
            # https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/PolyLine/
            # https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/LinearCellDemo/
            # The procedure to add lines is:
            #    vtkPoints()  ---------------------+> vtkPolyData()
            #    vtkPolyLine() -> vtkCellArray()  /
            #   then
            #    vtkPolyData() -> vtkPolyDataMapper() -> vtkActor() -> 
            #         vtkRenderer()

            line_width = obj_conf['line_width']
            if line_width == 'auto':
                if hasattr(self, 'screen_size'):
                    line_width = np.clip(self.screen_size[0]/1660 * 1.0,
                                         1.0, 2.0)
                else:
                    line_width = 1.0
            else:
                line_width = float(line_width)
            
            polyData = self.ConstructPolyData(raw_points, processes)

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polyData)
            actor = vtkActor()
            actor.SetMapper(mapper)
            self.actor = actor    # bind as early as possible, for properties
            self.property = actor.GetProperty()
            self.color = obj_conf['color']
            self.opacity = obj_conf.get('opacity', 1.0)
            self.line_width = line_width
            
            self.renderer.AddActor(actor)
            
            return self

        @staticmethod
        def LoadRawSwc(file_path):
            ntree = LoadSWCTree(file_path)
            processes = SplitSWCTree(ntree)
            ntree, processes = SWCDFSSort(ntree, processes)
            
            raw_points = ntree[1][:,0:3].astype(dtype_coor, copy=False)

            return processes, raw_points, ntree

        @staticmethod
        def ConstructPolyData(raw_points, processes):
            points = vtkPoints()
            #points.SetData( numpy_to_vtk(raw_points, deep=True) )
            # A tricky (dangerous) part:
            # 1. if raw_points is not continuous (raw_points.flags.contiguous)
            #    numpy_to_vtk() will create a continuous copy of it.
            # 2. the resulting vtk array will have a reference to the
            #    newly continued array or original array if it continuous.
            points.SetData( numpy_to_vtk(raw_points) )
            
            cells = vtkCellArray()
            # TODO: may try pre-allocate
            for proc in processes:
                polyLine = vtkPolyLine()
                lpid = polyLine.GetPointIds()
                # TODO: may try lpid.SetArray(proc.data, len(proc), false)
                lpid.SetNumberOfIds(len(proc))
                for i in range(0, len(proc)):
                    lpid.SetId(i, proc[i])
                cells.InsertNextCell(polyLine)

            polyData = vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(cells)
            return polyData

        def PopRawPoints(self):
            a = self.raw_points.T
            self.raw_points = None    # detach
            return a
        
        def ProcessColoring(self, scalar_color = None, max_depth = None, lut = None):
            """
            Color each neuronal process by scalar_color (real number in 0~1).
              s.ProcessColoring([0.1, 0.3, ...])
            Alternatively color the neuron by depth:
              s.ProcessColoring(max_depth = 31)

            e.g.
              from neu3dviewer.data_loader import *
              s = swcs[0]
              s.ProcessColoring(SimplifyTreeWithDepth(SplitSWCTree(s.tree_swc))[1:,2]/31.0)
            """
            mapper    = self.actor.GetMapper()     # vtkPolyDataMapper
            poly_data = mapper.GetInput()          # vtkPolyData
            seg_data  = poly_data.GetCellData()    # vtkCellData
            n_seg = poly_data.GetNumberOfCells()

            # case of removing colorings
            if (scalar_color is None) and (max_depth is None) \
                    and (lut is None):
                #mapper.SetScalarVisibility(False)
                seg_data.SetScalars(None)
                return

            # set LUT
            if (lut is None) and (self._color_lut is None):
                # use default LUT
                self.color_lut = self.gui_ctrl.GetDefaultSWCLUT()
            elif lut is not None:
                self.color_lut = lut

            # auto set scalar_color
            if isinstance(scalar_color, str) and (scalar_color == 'random'):
                # set random coloring
                n_color = 30
                scalar_color = (randint(0,n_color, (n_seg,)) + 0.5) / n_color
            elif isinstance(max_depth, numbers.Real):
                # coloring by depth, ignore scalar_color
                #ps = SplitSWCTree(self.tree_swc)
                ps = self.processes
                d  = SimplifyTreeWithDepth(ps, 'depth')
                scalar_color = (d - 0.5) / max_depth

            assert len(scalar_color) == n_seg
            seg_data.SetScalars(numpy_to_vtk(scalar_color, deep=True))
            #mapper.SetScalarVisibility(True)

        @property
        def color_lut(self):
            return self._color_lut

        @color_lut.setter
        def color_lut(self, lut):
            mapper = self.actor.GetMapper()     # vtkPolyDataMapper
            mapper.SetLookupTable(lut)          # lut can be None (default?)
            mapper.SetColorModeToMapScalars()   # use lookup table
            #mapper.SetScalarModeToUseCellData()  # the scalar data is color
            mapper.Modified()
            self._color_lut = lut
        
        @property
        def n_segments(self):
            return self.actor.GetMapper().GetInput().GetNumberOfCells()

        @property
        def visible(self):
            """Get/Set visibility of the whole fiber."""
            return self._visible
        
        @visible.setter
        def visible(self, v):
            if self._visible == v:
                return
            # How to hide a specific actor in python-vtk
            # https://stackoverflow.com/questions/69974435/how-to-hide-a-specific-actor-in-python-vtk
            # actor.GetProperty().SetOpacity(0)
            # Or
            # actor.VisibilityOff()
            # https://vtk.org/doc/nightly/html/classvtkProp.html#a03b15f78c7fce9041ddd91357c9c27ad
            if v:
                self.actor.VisibilityOn()
            else:
                self.actor.VisibilityOff()
            #Note: SetVisibility seems not work
            #if visible:
            #    self.actor.GetProperty().SetOpacity(1)
            #else:
            #    self.actor.GetProperty().SetOpacity(0)
            self._visible = v

        @property
        def color(self):
            """Get/Set color of the whole fiber."""
            return self._color

        @color.setter
        def color(self, c_desc):
            c3 = vtkGetColorAny3d(c_desc)
            # note: vtkProperty.SetColor() accept only RGB color, no alpha
            self.property.SetColor(c3)
            self._color = c_desc

        @property
        def opacity(self):
            """Get/Set opacity of the whole fiber."""
            return self._opacity

        @opacity.setter
        def opacity(self, op):
            self.property.SetOpacity(op)
            self._opacity = op

        @property
        def line_width(self):
            """Get/Set line width of the whole fiber."""
            return self._line_width

        @line_width.setter
        def line_width(self, lw):
            self.property.SetLineWidth(lw)
            self._line_width = lw

    class obj_mesh(TranslatorUnit):
        """
        prototype:
        "mesh": {
            "type": "mesh",
            "file_path": "path_to_file",  # can be .obj or .stl
            "color": "grey",
            "opacity": 0.7,
        },
        """

        @staticmethod
        def add_argument_to(parser):
            group = parser.add_argument_group('Mesh file options')
            group.add_argument('--mesh_path', action='append',
                    help='Read and draw OBJ/STL file path.')
            group.add_argument('--mesh_dir',
                    help='Read and draw OBJ/STL in directory.')
            group.add_argument('--mesh_color', metavar='COLOR',
                    help='Set color, like "Red".')
            group.add_argument('--mesh_opacity', metavar='OPACITY', type=float,
                    help='Set opacity, e.g. "0.7".')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if ('mesh_path' not in obj_desc) and ('mesh_dir' not in obj_desc):
                return None

            if 'mesh_path' not in obj_desc:  # for mesh_dir
                obj_desc['mesh_path'] = []

            if 'mesh_dir' in obj_desc:
                mesh_dir = obj_desc['mesh_dir']
                if os.path.isdir(mesh_dir):
                    # note down *.mesh_path files it to obj_desc['mesh_path']
                    import glob
                    fns = glob.glob(mesh_dir + '/*.obj')
                    fns.extend(glob.glob(mesh_dir + '/*.stl'))
                    if len(fns) == 0:
                        dbg_print(2, f'Option --mesh_dir "{mesh_dir}" contains no OBJ/STL.')
                    obj_desc['mesh_path'].extend(fns)
                else:
                    dbg_print(1, f'Option --mesh_dir "{mesh_dir}" not a directory.')

            file_paths = obj_desc['mesh_path']
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            
            obj_conf_s = [
                {
                    "type": "mesh",
                    "file_path": p,
                    "color": obj_desc.get('mesh_color', 'grey'),
                    "opacity": obj_desc.get('mesh_opacity', 0.7),
                }
                for p in file_paths
            ]
            return obj_conf_s

        def parse(self, obj_conf):
            file_path = obj_conf['file_path']
            if file_path.endswith('.obj'):
                reader = vtkOBJReader()
                reader.SetFileName(file_path)
            elif file_path.endswith('.stl'):
                reader = vtkSTLReader()
                reader.SetFileName(file_path)
            else:
                dbg_print(1, 'Unrecognized file format.')
                return self

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            self.actor = actor
            prop = actor.GetProperty()
            self.property = prop
            self._opacity = obj_conf['opacity']
            self._color   = obj_conf['color']
            if self._opacity == 1.0:
                prop.SetDiffuse(0.8)
                prop.SetDiffuseColor(vtkGetColorAny3d(self._color))
                prop.SetSpecular(0.3)
                prop.SetSpecularPower(60.0)
            else:
                prop.SetOpacity(self._opacity)
                prop.SetColor(vtkGetColorAny3d(self._color))
                #prop.SetAmbientColor(0.0, 0.0, 0.0)
                #prop.SetDiffuseColor(0.0, 0.0, 0.0)
                #prop.SetSpecularColor(0.0, 0.0, 0.0)

            self.renderer.AddActor(actor)
            return self


    class obj_AxesActor(TranslatorUnit):
        """
        prototype:
        "axes": {
            "type": "AxesActor",
            "ShowAxisLabels": False,
            "length": [100,100,100],
            "renderer": "0"
        },
        """
        def parse(self, obj_conf):
            # Create Axes object to indicate the orientation
            # vtkCubeAxesActor()
            # https://kitware.github.io/vtk-examples/site/Python/Visualization/CubeAxesActor/

            # Dynamically change position of Axes
            # https://discourse.vtk.org/t/dynamically-change-position-of-axes/691
            # Method 1
            axes = vtkAxesActor()
            axes.SetTotalLength(obj_conf.get('length', [1.0, 1.0, 1.0]))
            axes.SetAxisLabels(obj_conf.get('ShowAxisLabels', False))

            self.renderer.AddActor(axes)
            self.actor = axes
            return self

    class obj_Sphere(TranslatorUnit):
        """
        prototype:
        "3d_cursor": {
            "type": "Sphere",
        },
        """
        def parse(self, obj_conf):
            colors = vtkNamedColors()

            sphereSource = vtkSphereSource()
            sphereSource.SetCenter(0.0, 0.0, 0.0)
            sphereSource.SetRadius(2)
            sphereSource.SetPhiResolution(30)
            sphereSource.SetThetaResolution(30)
            
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            
            actor = vtkActor()
            actor.GetProperty().SetColor(colors.GetColor3d('Peacock'))
            actor.GetProperty().SetSpecular(0.6)
            actor.GetProperty().SetSpecularPower(30)
            actor.SetMapper(mapper)
            
            self.renderer.AddActor(actor)
            self.actor = actor
            return self

    class obj_OrientationMarker(TranslatorUnit):
        """
        prototype:
        "orientation": {
            "type": "OrientationMarker",
            "ShowAxisLabels": False,
            "renderer": "0"
        },
        """
        def parse(self, obj_conf):
            # Method 2
            # Ref: https://kitware.github.io/vtk-examples/site/Python/Interaction/CallBack/
            axes = vtkAxesActor()
            axes.SetTotalLength([1.0, 1.0, 1.0])
            axes.SetAxisLabels(obj_conf.get('ShowAxisLabels', False))
            axes.SetAxisLabels(True)

            # Ref: https://vtk.org/doc/nightly/html/classvtkOrientationMarkerWidget.html
            om = vtkOrientationMarkerWidget()
            om.SetOrientationMarker(axes)
            om.SetInteractor(self.gui_ctrl.interactor)
            om.SetDefaultRenderer(self.renderer)
            om.EnabledOn()
            om.SetInteractive(False)
            #om.InteractiveOn()
            om.SetViewport(0, 0, 0.2, 0.2)
            # TODO: the vtkOrientationMarkerWidget and RepeatingTimerHandler can cause program lose responds or Segmentation fault, for unknown reason.

            self.actor = om
            return self

    class obj_TextBox(TranslatorUnit):
        """
        prototype:
        "message": {
            "type": "TextBox",
            "text": GenerateKeyBindingDoc(),
            "font_size": "auto",                   # auto or a number
            "font_family": "mono",                 # optional
            "color": [1.0, 1.0, 0.1],
            "background_color": [0.0, 0.2, 0.5],
            "background_opacity": 0.8,             # optional
            "position": "center",                  # optional
            "frame_on": True,                      # optional
        },
        """
        def parse(self, obj_conf):
            text_actor = vtkTextActor()
            text_actor.SetInput(obj_conf['text'])
            self.actor = text_actor
            self._text = obj_conf['text']

            prop = text_actor.GetTextProperty()
            self.property = prop
            # set font
            if obj_conf.get('font_family', '') == 'mono':
                prop.SetFontFamilyToCourier()
            self._font_size = obj_conf['font_size']
            # set color of text and background
            prop.SetColor(obj_conf['color'])
            prop.SetBackgroundColor(obj_conf['background_color'])
            prop.SetBackgroundOpacity(obj_conf.get('background_opacity', 1.0))
            # set frame (bounding box)
            if obj_conf.get('frame_on', False):
                prop.FrameOn()

            # set position of text
            position = obj_conf.get('position', None)
            self._position = position
            self.OnWindowResize(self.gui_ctrl.render_window, None)

            self.renderer.AddActor2D(text_actor)
            self._win_resize_ob_tag = \
                self.gui_ctrl.render_window. \
                AddObserver(vtkCommand.WindowResizeEvent, self.OnWindowResize)
            return self
    
        def __del__(self):
            if hasattr(self, '_win_resize_ob_tag'):
                self.gui_ctrl.render_window. \
                    RemoveObserver(self._win_resize_ob_tag)
            super().__del__()
    
        def OnWindowResize(self, obj, event):
            # assert isinstance(obj, vtkRenderWindow)
            win_size = obj.GetSize()
            if event is not None:
                dbg_print(5, "obj_text: window size:", win_size)

            text_actor = self.actor
            prop = self.property

            if text_actor is None:
                return

            # set auto font size
            if self._font_size == 'auto':
                screen_size = obj.GetScreenSize()
                #dpi = self.gui_ctrl.render_window.dpi
                #fs = int(dpi / 96 * 31 * screen_size[1] / 1600)
                fs = int(26 * screen_size[1] / 1600)
            else:
                fs = self._font_size
            if prop.GetFontSize() != fs:
                dbg_print(4, 'obj_text: set font size:', fs)
                prop.SetFontSize(fs)

            # set auto position
            position = self._position
            if position == 'center':
                prop.SetVerticalJustificationToCentered()
                v = _a([0]*4)
                text_actor.GetBoundingBox(self.renderer, v)
                text_actor.SetPosition(win_size[0]/2 - (v[1]-v[0])/2, win_size[1]/2)
            elif position == 'lowerright':
                #prop.SetJustificationToRight()
                prop.SetVerticalJustificationToBottom()
                v = _a([0]*4)
                text_actor.GetBoundingBox(self.renderer, v)
                text_actor.SetPosition(win_size[0] - (v[1]-v[0]), 0)
            elif isinstance(position, (list, tuple)):  # assume position is [x,y]
                text_actor.SetPosition(position[0], position[1])
    
        @property
        def text(self):
            return self._text
    
        @text.setter
        def text(self, msg):
            self.actor.SetInput(msg)
            # re-position
            self.OnWindowResize(self.gui_ctrl.render_window, None)
            self._text = msg

    class obj_Background(TranslatorUnit):
        """
        prototype:
        "background": {
            "type": "Background",
            "color": "Black"
#            "color": "Wheat"
        },
        """
        def parse(self, obj_conf):
            colors = vtkNamedColors()
            self.renderer.SetBackground(colors.GetColor3d(obj_conf['color']))
            return self

        def remove(self, rm_conf = {}):
            dbg_print(1, "we don't remove background.")

    class obj_Camera(TranslatorUnit):
        """
        prototype:
        "camera1": {
            "type": "Camera",
            "renderer": "0",
            "SetPosition": [x, y, z],
            "SetFocalPoint": [x, y, z],
            "SetViewUp": [x, y, z],
            "SetViewAngle": [x, y, z],
            "Azimuth": 45,
            "Elevation": 30,
            "clipping_range": [0.0001, 100000]
        },
        Or:
        "camera2": {
            "type": "Camera",
            "renderer": "1",
            "follow_direction": "camera1"
        },
        """
        def parse(self, obj_conf):
            if 'renderer' in obj_conf:
                if obj_conf.get('new', False) == False:
                    cam = self.renderer.GetActiveCamera()
                    self.renderer.ResetCameraClippingRange()
                    self.renderer.ResetCamera()
                else:
                    cam = self.renderer.MakeCamera()
            else:
                cam = vtkCamera()

            if 'clipping_range' in obj_conf:
                cam.SetClippingRange(obj_conf['clipping_range'])

            item_name = {
                'Set':['Position', 'FocalPoint', 'ViewUp', 'ViewAngle'],
                ''   :['Azimuth', 'Elevation']
            }
            for f_prefix, its in item_name.items():
                for it in its:
                    if it in obj_conf:
                        getattr(cam, f_prefix + it)(obj_conf[it])

            if 'follow_direction' in obj_conf:
                cam_ref = self.gui_ctrl.scene_objects[
                              obj_conf['follow_direction']].actor
                cam.DeepCopy(cam_ref)
                cam.SetClippingRange(0.1, 1000)
                AlignCameraDirection(cam, cam_ref)

                CameraFollowCallbackFunction.cam1 = cam_ref
                CameraFollowCallbackFunction.cam2 = cam

                cam_ref.AddObserver( \
                    'ModifiedEvent', CameraFollowCallbackFunction)

            self.actor = cam

            return self

        def remove(self, rm_conf = {}):
            dbg_print(1, "we don't usually remove camera.")

    class animation_rotation(TranslatorUnit):
        """
        prototype:
        {
            'type'           : 'rotation',
            'time'           : 6.0,
            'fps'            : 60.0,
            'degree_per_sec' : 60.0,
            'save_pic_path'  : 'pic_tmp/haha_t=%06.4f.png',
        }
        """

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--animation', type=int, metavar='TYPE_INT',
                    help= \
            """Run an animation and exit.
            Currently, only rotation (TYPE 1) is supported.
            Usually, you would use it like:
              --off_screen_rendering 1 --animation 1
            """
            )
            
        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            if ('animation' not in cmd_obj_desc) or \
               (cmd_obj_desc['animation'] == 0):
                return None

            cg_conf = {
                'type'           : 'rotation',
                'time'           : 6.0,
                'fps'            : 60.0,
                'degree_per_sec' : 60.0,
                'save_pic_path'  : 'pic_tmp/haha_t=%06.4f.png',
            }
            return cg_conf

        def parse(self, cg_conf):
            if not cg_conf: return
            time.sleep(1.0)         # to prevent bug (let fully init)

            save_pic_path = cg_conf['save_pic_path']

            obj = self.gui_ctrl.interactor   # iren
            event = ''
            cam1 = self.gui_ctrl.GetMainRenderer().GetActiveCamera()

            rotator = execSmoothRotation(cam1, cg_conf['degree_per_sec'])
            rotator.startat(0)
            for k in range(int(cg_conf['fps'] * cg_conf['time'])):
                t_now = 1.0/cg_conf['fps'] * k;
                rotator(obj, event, t_now)
                if save_pic_path:
                    self.gui_ctrl.ShotScreen( \
                        save_pic_path % (t_now))

    class animation_take_shot(TranslatorUnit):
        """
        prototype:
        {
            'type'           : 'take_shot',
            'delay'          : 1.0,
            'save_pic_path'  : 'pic_tmp/haha_t=%06.4f.png',
        }
        """

        @staticmethod
        def add_argument_to(parser):
            # usually, you would use it like
            # --off_screen_rendering 1 --screenshot a.png
            #Or
            # --screenshot a.png --no_interaction 1
            parser.add_argument('--screenshot', metavar='PNG_PATH',
                                nargs='?', const = '',
                    help='Take a screen shot and save to the path.')

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            if 'screenshot' not in cmd_obj_desc:
                return None
            
            pic_path = 'screenshot_' \
                       + WindowsFriendlyDateTime() \
                       + '.png'

            if cmd_obj_desc['screenshot']:
                pic_path = cmd_obj_desc['screenshot']
                if not pic_path.endswith('.png'):
                    pic_path += '.png'

            cg_conf = {
                'type'           : 'take_shot',
                'delay'          : 1.0,
                'save_pic_path'  : pic_path,
            }
            
            return cg_conf

        def parse(self, cg_conf):
            if not cg_conf: return
            time.sleep(cg_conf['delay'])  # to prevent bug (let fully init)

            save_pic_path = cg_conf['save_pic_path']

            if save_pic_path:
                self.gui_ctrl.ShotScreen(save_pic_path)
 
            dbg_print(4, 'Screenshot saved to', save_pic_path)

