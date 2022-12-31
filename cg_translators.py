# Computer graphics translators.

import numpy as np

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
    dbg_print
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

    class TranslatorUnit:
        cmd_line_opt = ''

        def __init__(self, gui_ctrl, renderer):
            self.gui_ctrl = gui_ctrl
            self.renderer = renderer

        def parse(self, st):
            """
            translate json discription to obj on screen.
            """
            pass

    class prop_volume(TranslatorUnit):
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
        obj_conf_type = 'volume'
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
            
            return volume

    class obj_swc(TranslatorUnit):
        obj_conf_type = 'swc'
        def parse(self, obj_conf):
            ntree = LoadSWCTree(obj_conf['file_path'])
            processes = SplitSWCTree(ntree)
            
            self.gui_ctrl.point_graph = GetUndirectedGraph(ntree)
            raw_points = ntree[1][:,0:3]
            self.gui_ctrl.point_set_holder.AddPoints(raw_points.T, '')
            
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
            #actor.raw_points = raw_points  # for convenience

            return actor
    class obj_AxesActor(TranslatorUnit):
        obj_conf_type = 'AxesActor'
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
            return axes

    class obj_Sphere(TranslatorUnit):
        obj_conf_type = 'Sphere'
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
            return actor

    class obj_OrientationMarker(TranslatorUnit):
        obj_conf_type = 'OrientationMarker'
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

            return om
    class obj_Background(TranslatorUnit):
        obj_conf_type = 'Background'
        def parse(self, obj_conf):
            colors = vtkNamedColors()
            self.renderer.SetBackground(colors.GetColor3d(obj_conf['color']))
            return self.renderer

    class obj_Camera(TranslatorUnit):
        obj_conf_type = 'Camera'
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
                              obj_conf['follow_direction']]
                cam.DeepCopy(cam_ref)
                cam.SetClippingRange(0.1, 1000)
                AlignCameraDirection(cam, cam_ref)

                CameraFollowCallbackFunction.cam1 = cam_ref
                CameraFollowCallbackFunction.cam2 = cam

                cam_ref.AddObserver( \
                    'ModifiedEvent', CameraFollowCallbackFunction)

            return cam

