# Computer graphics translators, the real workhorse.
# You may also call them operators (of VTK), but "translator" is descriptive stype, "operator" is imperative style, I prefer the former, so everything here is passive and the control point is left in the main code (GUIControl).

# Ref. for VTK
# Python Wrappers for VTK
# https://vtk.org/doc/nightly/html/md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html

# Demonstrates physically based rendering using image based lighting and a skybox.
# https://kitware.github.io/vtk-examples/site/Python/Rendering/PBR_Skybox/

import datetime
import numpy as np
from numpy import array as _a
import json

from vtkmodules.vtkCommonCore import (
    vtkPoints,
    VTK_CUBIC_INTERPOLATION
)
from vtkmodules.vtkCommonDataModel import (
    vtkPiecewiseFunction,
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
    vtkColorSeries
)
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
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper

from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.util.numpy_support import numpy_to_vtk

from utils import (
    GetNonconflitName,
    vtkGetColorAny,
    _mat3d,
    dbg_print,
    UpdatePropertyOTFScale,
    UpdatePropertyCTFScale,
    GetColorScale,
    SetColorScale,
)

from ui_interactions import (
    MyInteractorStyle,
    execSmoothRotation,
)

from data_loader import *

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
    The collection of translators to convert json discription to computer 
    graphic objects.
    The ideal is that eventually GUIControl do not contain any
    implimentation details.
    All the translator units should not have state.
    Also handle commandline parse.
    External code should be easy to modify this class to extend its function.
    """

    class TranslatorUnit:
        def __init__(self, gui_ctrl, renderer):
            self.gui_ctrl = gui_ctrl
            self.renderer = renderer
            self.actor = None
            self.position = None

        def parse(self, st):
            """
            translate json discription to obj on screen.
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

        #@staticmethod
        #def add_argument_to(args_parser):
        #    pass

        #@staticmethod
        #def parse_cmd_args(cmd_obj_desc):
        #    pass

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

    class init_window(TranslatorUnit):
        """
        prototype:
        "window": {
            "size": [2400, 1800],
            "title": "SimpleRayCast",
            "number_of_layers": 2,
#            "stereo_type": "SplitViewportHorizontal"
        },
        """
        def parse(self, win_conf):
            # TODO: should we stop the old window?
            # TODO: try vtkVRRenderWindow?
            if self.gui_ctrl.render_window is None:
                self.gui_ctrl.render_window = vtkRenderWindow()
            render_window = self.gui_ctrl.render_window
            if 'size' in win_conf:
                render_window.SetSize(win_conf['size'])
            if 'title' in win_conf:
                render_window.SetWindowName(win_conf['title'])
            if 'number_of_layers' in win_conf:
                render_window.SetNumberOfLayers(
                    win_conf['number_of_layers'])
            if 'stereo_type' in win_conf:
                render_window.StereoCapableWindowOn()
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

        def parse_post_renderers(self, win_conf):
            # Off screen rendering
            # https://discourse.vtk.org/t/status-of-vtk-9-0-with-respect-to-off-screen-rendering-under-ubuntu-with-pip-install/5631/2
            # TODO: add an option for off screen rendering
            if win_conf.get('off_screen_rendering', False):
                self.gui_ctrl.render_window.SetOffScreenRendering(1)
                self.gui_ctrl.do_not_start_interaction = True
            else:
                self.gui_ctrl.do_not_start_interaction = False
            # Hint: you may use xdotoll to control an off-screen program
            # e.g. xdotool search --name SimpleRayCast key 'q'

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--off_screen_rendering',
                    type=int, choices=[0, 1],
                    help='Enable off-screen rendering. 1=Enable, 0=Disable.')
            parser.add_argument('--stereo_type',
                    help=
"""
Enable stereo rendering, set the type here.
Possible types:
  CrystalEyes, RedBlue, Interlaced, Left, Right, Fake, Emulate,
  Dresden, Anaglyph, Checkerboard, SplitViewportHorizontal
""")

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            win_conf = {}
            if 'off_screen_rendering' in cmd_obj_desc:
                win_conf.update({
                    'off_screen_rendering': cmd_obj_desc['off_screen_rendering'] > 0
                })
            if 'stereo_type' in cmd_obj_desc:
                win_conf.update({
                    'stereo_type': cmd_obj_desc['stereo_type']
                })
            return win_conf

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
                renderers[key] = renderer
                # add new renderer to window
                render_window.AddRenderer(renderer)

    class init_interactor(TranslatorUnit):
        def parse(self, ui_conf):
            if (not ui_conf) and (self.gui_ctrl.interactor):
                # it is initialized, so do not change anything
                return self.gui_ctrl.interactor
            # Create the interactor (for keyboard and mouse)
            interactor = vtkRenderWindowInteractor()
            interactor.SetInteractorStyle(
                    MyInteractorStyle(interactor, self.gui_ctrl))
        #    interactor.AddObserver('ModifiedEvent', ModifiedCallbackFunction)
            interactor.SetRenderWindow(self.gui_ctrl.render_window)
            return interactor

    class init_scene(TranslatorUnit):
        """
        {
            "object_properties": {...},
            "objects": {}
        }
        """
        def parse(self, obj_conf):
            pass

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--scene',
                    help='Project scene file path. e.g. for batch object loading.')

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            if 'scene' in cmd_obj_desc:
                # TODO: maybe move this before init of gui, and pass it as init param.
                scene_ext = json.loads(open(cmd_obj_desc['scene']).read())
            else:
                scene_ext = {}
            return scene_ext

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
            
            if 'copy_from' in prop_conf:
                dbg_print(4, 'Copy propperty from', prop_conf['copy_from'])
                # construct a volume property by copying from exist
                ref_prop = self.gui_ctrl.object_properties[
                               prop_conf['copy_from']]
                volume_property.DeepCopy(ref_prop)
                volume_property.prop_conf = prop_conf
                volume_property.ref_prop = ref_prop
                self.modify(volume_property, prop_conf)
                return volume_property

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
        }
        """
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
            ref_prop_conf = obj_conf.get('property', 'volume')
            if isinstance(ref_prop_conf, dict):
                # add new property
                prop_name = self.gui_ctrl.GetNonconflitName('volume', 'property')
                dbg_print(3, 'AddObject(): Adding prop:', prop_name)
                self.gui_ctrl.AddObjectProperty(prop_name, ref_prop_conf)
                volume_property = self.gui_ctrl.object_properties[prop_name]
            else:
                dbg_print(3, 'AddObject(): Using existing prop:', ref_prop_conf)
                volume_property = self.gui_ctrl.object_properties[ref_prop_conf]

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

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--filepath',
                    help='image stack filepath')
            parser.add_argument('--level',
                    help='for multi-level image (.ims), load only that level')
            parser.add_argument('--channel',
                    help='Select channel for IMS image.')
            parser.add_argument('--time_point',
                    help='Select time point for IMS image.')
            parser.add_argument('--range',
                    help='Select range within image.')
            parser.add_argument('--colorscale',
                    help='Set scale of color transfer function.')
            parser.add_argument('--origin',
                    help='Set origin of the volume.')
            parser.add_argument('--rotation_matrix',
                    help='Set rotation matrix of the volume.')
            parser.add_argument('--oblique_image',
                    help='Overwrite the guess of if the image is imaged oblique.')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if 'filepath' not in obj_desc:
                return None

            file_path = obj_desc['filepath']
            if file_path.endswith('.tif'):
                # assume this a volume
                obj_conf = {
                    "type": "volume",
                    "view_point": "auto",
                    "file_path": file_path
                }
            elif file_path.endswith('.ims') or file_path.endswith('.h5'):
                # assume this a IMS volume
                obj_conf = {
                    "type": "volume",
                    "view_point": "auto",
                    "file_path": file_path,
                    "level": obj_desc.get('level', '0'),
                    "channel": obj_desc.get('channel', '0'),
                    "time_point": obj_desc.get('time_point', '0'),
                    "range": obj_desc.get('range', '[:,:,:]')
                }
            else:
                dbg_print(1, 'Unreconized source format.')
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
                    'copy_from': 'volume',
                    'type'     : 'volume',
                    'opacity_transfer_function': {'opacity_scale': s},
                    'color_transfer_function'  : {'trans_scale': s}
                }})
            else:
                obj_conf.update({'property': 'volume'})

            return obj_conf

    class obj_lychnis_blocks(TranslatorUnit):
        """
        prototype:
        {
            "type"     : "lychnis_blocks",
            "file_path": obj_desc['lychnis_blocks'],
        }
        """
        #TODO: we might return ourself instead of volume_loader
        def parse(self, obj_conf):
            self.gui_ctrl.volume_loader.ImportLychnixVolume( \
                obj_conf['file_path'])
            return self.gui_ctrl.volume_loader
        
        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--lychnis_blocks',
                    help='Path of lychnix blocks.json')

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

    class obj_swc(TranslatorUnit):
        """
        prototype:
        "swc": {
            "type": "swc",
            "color": "Tomato",
            "linewidth": 2.0,
            "file_path": "RM006-004-lychnis/F5.json.swc"
        }
        """
        def parse(self, obj_conf):
            ntree = LoadSWCTree(obj_conf['file_path'])
            processes = SplitSWCTree(ntree)
            
            self.gui_ctrl.point_graph = GetUndirectedGraph(ntree)
            raw_points = ntree[1][:,0:3]
            self.raw_points = raw_points
            
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
            points.SetData( numpy_to_vtk(raw_points, deep=True) )
            
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

            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polyData)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(
                vtkGetColorAny(obj_conf['color']))
            actor.GetProperty().SetLineWidth(obj_conf.get('linewidth', 2.0))
            self.renderer.AddActor(actor)
            self.actor = actor
            return self

        def PopRawPoints(self):
            a = self.raw_points.T
            self.raw_points = None    # detach
            return a

        # TODO: convert it to property
        def SetVisibility(self, visible):
            # How to hide a specific actor in python-vtk
            # https://stackoverflow.com/questions/69974435/how-to-hide-a-specific-actor-in-python-vtk
            # actor.GetProperty().SetOpacity(0)
            # Or
            # actor.VisibilityOff()
            # https://vtk.org/doc/nightly/html/classvtkProp.html#a03b15f78c7fce9041ddd91357c9c27ad
            if visible:
                self.actor.VisibilityOn()
            else:
                self.actor.VisibilityOff()

            #if visible:
            #    self.actor.GetProperty().SetOpacity(1)
            #else:
            #    self.actor.GetProperty().SetOpacity(0)

        @staticmethod
        def add_argument_to(parser):
            parser.add_argument('--swc', action='append',
                    help='Read and draw swc file.')
            parser.add_argument('--swc_dir',
                    help='Read and draw swc files in the directory.')
            parser.add_argument('--fibercolor',
                    help='Set fiber color.')

        @staticmethod
        def parse_cmd_args(obj_desc):
            if ('swc' not in obj_desc) and ('swc_dir' not in obj_desc):
                return None

            if 'swc_dir' in obj_desc:
                # note down *.swc files it to obj_desc['swc']
                import glob
                fns = glob.glob(obj_desc['swc_dir'] + '/*.swc')
                if 'swc' not in obj_desc:
                    obj_desc['swc'] = []
                obj_desc['swc'].extend(fns)
        
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
                    "color": obj_desc.get('fibercolor', c),
                    "file_path": obj_desc['swc'][id_s]
                }
                li_obj_conf.append(obj_conf)

            return li_obj_conf

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
            # TODO: the vtkOrientationMarkerWidget and RepeatingTimerHandler can cause program lose respons or Segmentation fault, for unknown reason.

            self.actor = om
            return self

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

        @staticmethod
        def add_argument_to(parser):
            # usually, you would use it like
            # --off_screen_rendering 1 --animation 1
            parser.add_argument('--animation', type=int,
                    help='Run an off-screen animation and exit.')

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

    class animation_take_shot(TranslatorUnit):
        """
        prototype:
        {
            'type'           : 'take_shot',
            'delay'          : 1.0,
            'save_pic_path'  : 'pic_tmp/haha_t=%06.4f.png',
            'no_interaction' : False
        }
        """
        def parse(self, cg_conf):
            if not cg_conf: return
            time.sleep(cg_conf['delay'])  # to prevent bug (let fully init)

            save_pic_path = cg_conf['save_pic_path']

            if save_pic_path:
                self.gui_ctrl.ShotScreen(save_pic_path)
 
            dbg_print(4, 'Screenshot saved to', save_pic_path)

            self.gui_ctrl.do_not_start_interaction = cg_conf['no_interaction']

        @staticmethod
        def add_argument_to(parser):
            # usually, you would use it like
            # --off_screen_rendering 1 --save_screen a.png
            #Or
            # --save_screen a.png --no_interaction 1
            parser.add_argument('--save_screen',
                    help='Take a screen shot and save to the path.')
            parser.add_argument('--no_interaction', type=int, choices=[0, 1],
                    help='Exit after the screen shot.')

        @staticmethod
        def parse_cmd_args(cmd_obj_desc):
            if 'save_screen' not in cmd_obj_desc:
                return None
            
            pic_path = 'pic_tmp/a_' \
                       + str(datetime.datetime.now()).replace(' ', '_') \
                       + '.png'

            if cmd_obj_desc['save_screen']:
                pic_path = cmd_obj_desc['save_screen']

            cg_conf = {
                'type'           : 'take_shot',
                'delay'          : 1.0,
                'save_pic_path'  : pic_path,
                'no_interaction' : cmd_obj_desc.get('no_interaction', False)>0
            }
            
            return cg_conf

