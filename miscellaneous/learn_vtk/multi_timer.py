#!/usr/bin/env python3

import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle

from vtkmodules.vtkRenderingCore import (
vtkRenderWindow,
vtkRenderWindowInteractor,
vtkRenderer,
vtkActor,
vtkPolyDataMapper,
)

from vtkmodules.vtkFiltersSources import vtkSphereSource

def TimerCallback(caller, event):
    print(event)
    caller.DestroyTimer(TimerCallback.timer_id)

if __name__ == '__main__':
    renderer = vtkRenderer()

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    sphereSource = vtkSphereSource()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphereSource.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer.AddActor(actor)
    renderer.ResetCameraClippingRange()
    renderer.ResetCamera()
    render_window.Render()

    interactor.Initialize()
    
    interactor.AddObserver('TimerEvent', TimerCallback)
    TimerCallback.timer_id = interactor.CreateRepeatingTimer(250)
    
    render_window.Render()
    interactor.Start()
