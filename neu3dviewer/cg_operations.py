# SPDX-License-Identifier: GPL-3.0-or-later

# Core algorithms for operating the drawings.
# Might be called by ui_interaction.py

import numpy as np
from numpy import array as _a

from vtkmodules.vtkCommonCore import (
    vtkPoints,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
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
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter,
    vtkPropPicker,
    vtkPointPicker,
    vtkActor,
    vtkPolyDataMapper,
)
from vtkmodules.vtkFiltersCore import vtkClipPolyData
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette

# loading this consumes ~0.1 second!
# might move it to where it is used.
from vtkmodules.util.numpy_support import numpy_to_vtk

from .utils import (
    dbg_print,
)
from .data_loader import (
    GetUndirectedGraph,
)

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
          box_scaling          : the scale of the bounding box
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
        # The point graph is initialized when adding SWC type objects and used to find adjacent points
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
        self.global_pid = None
        self.volume_clipper = None
        self.isOn = False
        self.swc_polydata = None
        self.swc_mapper = None
        self.cut_swc_flag = True
        self.focus_swc = None
        self.last_swc_name = None

    def SetPointsInfo(self, point_graph, point_coor):
        self.point_searcher = PointSearcher(point_graph, points_coor=point_coor)

    def SetGUIController(self, gui_ctrl):
        # called in GUIController
        self.gui_ctrl = gui_ctrl
        self.renderer = gui_ctrl.GetMainRenderer()
        self.iren = gui_ctrl.interactor
        self.gui_ctrl.volume_observers.append(self)

    def InitPointSearcher(self, global_pid):
        obj_name = self.gui_ctrl.point_set_holder.GetNameByPointId(global_pid)
        if obj_name == self.last_swc_name:
            # we are updated
            return
        dbg_print(4, 'FocusModeController::InitPointSearcher(): swc =', obj_name)
        self.last_swc_name = obj_name
        if self.gui_ctrl.scene_saved['objects'][obj_name]['type'] != 'swc':
            dbg_print(3, 'FocusModeController::InitPointSearcher(): Not a swc object.')
            self.swc_mapper = None
            self.swc_polydata = None
            self.point_searcher = None
            return
        swc_obj = self.gui_ctrl.scene_objects[obj_name]
        if not hasattr(swc_obj, 'point_graph'):
            swc_obj.point_graph = GetUndirectedGraph(swc_obj.tree_swc)
            swc_obj.raw_points = swc_obj.tree_swc[1][:,0:3]
        self.swc_mapper = swc_obj.actor.GetMapper()
        self.swc_polydata = self.swc_mapper.GetInput()
        self.point_searcher = PointSearcher(
            swc_obj.point_graph,
            points_coor = swc_obj.raw_points.T)

    def SetCenterPoint(self, global_pid):
        """
        Update spot site according to point position.
        interface
        """
        self.global_pid = global_pid
        if not self.isOn:
            return
        self.InitPointSearcher(global_pid)
        local_pid = self.gui_ctrl.point_set_holder.GetLocalPid(global_pid)
        points = self.point_searcher \
                    .SearchPointsAround_coor(local_pid)
        if not self.volume_clipper:
            self.volume_clipper = VolumeClipper(points)
        else:
            self.volume_clipper.SetPoints(points)
        self.gui_ctrl.UpdateVolumesNear(
            self.point_searcher.points_coordinate.T[local_pid])
        self.volume_clipper.RestoreVolumes(self.renderer.GetVolumes())
        self.volume_clipper.CutVolumes(self.renderer.GetVolumes())
        if self.cut_swc_flag:
            if self.focus_swc:
                self.gui_ctrl.GetMainRenderer().RemoveActor(self.focus_swc)
            oldClipper = vtkClipPolyData()
            oldClipper.SetInputData(self.swc_polydata)
            oldClipper.SetClipFunction(self.volume_clipper.planes[0])
            path = self.point_searcher.SearchPathAround(local_pid)
            self.swc_mapper.SetInputData(oldClipper.GetOutput())
            self.CreateLines(path[1])
        self.iren.GetRenderWindow().Render()

    def Toggle(self):
        # interface
        if self.isOn:
            self.isOn = False
            self.volume_clipper.RestoreVolumes(self.renderer.GetVolumes())
            if self.cut_swc_flag:
                self.swc_mapper.SetInputData(self.swc_polydata)
                self.gui_ctrl.GetMainRenderer().RemoveActor(self.focus_swc)
            self.iren.GetRenderWindow().Render()
        else:
            self.isOn = True
            if self.global_pid:
                self.SetCenterPoint(self.global_pid)

    def CreateLines(self, path):
        points = vtkPoints()
        points.SetData(numpy_to_vtk(
            self.point_searcher.points_coordinate.T, deep=True))
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

