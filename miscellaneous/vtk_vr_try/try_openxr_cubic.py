# OpenXR test
# coordinate system ref.
#  VTK::RenderingVR
#   https://docs.vtk.org/en/latest/modules/vtk-modules/Rendering/VR/README.html

# set view direction and position
# class vtkRenderWindow.html
#  https://vtk.org/doc/nightly/html/classvtkRenderWindow.html#details

import time

import numpy
import vtk

from vtkmodules.util.numpy_support import vtk_to_numpy

ENABLE_VR = "OpenXR"

if ENABLE_VR == "OpenXR":
    XRenderer = vtk.vtkOpenXRRenderer
    XRenderWindow = vtk.vtkOpenXRRenderWindow
    XCamera = vtk.vtkOpenXRCamera
    XRenderWindowInteractor = vtk.vtkOpenXRRenderWindowInteractor
elif ENABLE_VR == "OpenVR":
    XRenderer = vtk.vtkOpenVRRenderer
    XRenderWindow = vtk.vtkOpenVRRenderWindow
    XCamera = vtk.vtkOpenVRCamera
    XRenderWindowInteractor = vtk.vtkOpenVRRenderWindowInteractor
else:
    XRenderer = vtk.vtkRenderer
    XRenderWindow = vtk.vtkRenderWindow
    XCamera = vtk.vtkCamera
    XRenderWindowInteractor = vtk.vtkRenderWindowInteractor

def main():
    renderWindow = XRenderWindow()
    renderer = XRenderer()
    cam = XCamera()
    renderer.SetActiveCamera(cam)
    renderWindow.AddRenderer(renderer)

    renderer.SetBackground(0.2, 0.3, 0.4)

    if 0:
        doll = vtk.vtkSphereSource()
        doll.SetPhiResolution(80)
        doll.SetThetaResolution(80)
        doll.SetRadius(100)
        doll.Update()
    else:  # cube
        doll = vtk.vtkCubeSource()
        doll.SetXLength(200)
        doll.SetYLength(200)
        doll.SetZLength(200)
        doll.Update()

    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputConnection(doll.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.2)

    renderer.AddActor(actor)

    renderer.ResetCamera()
    cam.SetPosition(0, 1000, 1000)
    cam.SetFocalPoint(0, 100, 0)
    #renderer.ResetCameraClippingRange()   # case problem
    renderer.Modified()
    print("Get dist", cam.GetDistance())
    #renderer.ResetCamera()

    iren = XRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)
    if ENABLE_VR:
        iren.SetActionManifestDirectory(r"./")

    iren.Initialize()
    if ENABLE_VR:
        iren.DoOneEvent(renderWindow, renderer)
        iren.DoOneEvent(renderWindow, renderer)  # Needed by monado so that it starts to render

    mat_p_P2W = vtk.vtkMatrix4x4()
    renderWindow.GetPhysicalToWorldMatrix(mat_p_P2W)
    #print('mat_p_P2W', mat_p_P2W)
    # convert to numpy matrix
    mat_p_P2W = numpy.array([[mat_p_P2W.GetElement(i, j) for j in range(4)] for i in range(4)])
    print('mat_p_P2W', mat_p_P2W)

    vec_p_view_direction = renderWindow.GetPhysicalViewDirection()
    print('vec_p_view_direction', vec_p_view_direction)

    vec_p_view_up = renderWindow.GetPhysicalViewUp()
    print('vec_p_view_up', vec_p_view_up)

    vec_p_translation = renderWindow.GetPhysicalTranslation()
    print('vec_p_translation', vec_p_translation)

    val_p_scale = renderWindow.GetPhysicalScale()
    print('val_p_scale', val_p_scale)

    renderWindow.SetPhysicalTranslation(0, 200, -500)

    renderWindow.Render()
    iren.Start()

    return 0

if __name__ == "__main__":
    main()