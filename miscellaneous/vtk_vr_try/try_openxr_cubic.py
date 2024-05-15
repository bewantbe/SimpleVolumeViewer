# OpenXR test
# coordinate system ref.
#  VTK::RenderingVR
#   https://docs.vtk.org/en/latest/modules/vtk-modules/Rendering/VR/README.html

# set view direction and position
# class vtkRenderWindow.html
#  https://vtk.org/doc/nightly/html/classvtkRenderWindow.html#details

import time

import numpy
#import vtk

from vtkmodules.vtkRenderingCore import (
    vtkActor,
)
from vtkmodules.vtkFiltersSources import (
    vtkSphereSource,
    vtkCubeSource
)
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
from vtkmodules.vtkCommonMath import (
    vtkMatrix4x4,
)

from vtkmodules.util.numpy_support import vtk_to_numpy

ENABLE_VR = "OpenXR"

if ENABLE_VR == "OpenXR":
    from vtkmodules.vtkRenderingOpenXR import (
        vtkOpenXRRenderer,
        vtkOpenXRRenderWindow,
        vtkOpenXRCamera,
        vtkOpenXRRenderWindowInteractor,
    )
    XRenderer = vtkOpenXRRenderer
    XRenderWindow = vtkOpenXRRenderWindow
    XCamera = vtkOpenXRCamera
    XRenderWindowInteractor = vtkOpenXRRenderWindowInteractor
elif ENABLE_VR == "OpenVR":
    from vtkmodules.vtkRenderingOpenVR import (
        vtkOpenVRRenderer,
        vtkOpenVRRenderWindow,
        vtkOpenVRCamera,
        vtkOpenVRRenderWindowInteractor,
    )
    XRenderer = vtkOpenVRRenderer
    XRenderWindow = vtkOpenVRRenderWindow
    XCamera = vtkOpenVRCamera
    XRenderWindowInteractor = vtkOpenVRRenderWindowInteractor
else:
    from vtkmodules.vtkRenderingCore import (
        vtkRenderer,
        vtkRenderWindow,
        vtkCamera,
        vtkRenderWindowInteractor,
    )
    XRenderer = vtkRenderer
    XRenderWindow = vtkRenderWindow
    XCamera = vtkCamera
    XRenderWindowInteractor = vtkRenderWindowInteractor

def main():
    renderWindow = XRenderWindow()
    renderer = XRenderer()
    cam = XCamera()
    renderer.SetActiveCamera(cam)
    renderWindow.AddRenderer(renderer)

    renderer.SetBackground(0.2, 0.3, 0.4)

    if 0:
        doll = vtkSphereSource()
        doll.SetPhiResolution(80)
        doll.SetThetaResolution(80)
        doll.SetRadius(100)
        doll.Update()
    else:  # cube
        doll = vtkCubeSource()
        doll.SetXLength(200)
        doll.SetYLength(200)
        doll.SetZLength(200)
        doll.Update()

    mapper = vtkOpenGLPolyDataMapper()
    mapper.SetInputConnection(doll.GetOutputPort())
    actor = vtkActor()
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

    mat_p_P2W = vtkMatrix4x4()
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