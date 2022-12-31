import time

import numpy as np
from numpy import sqrt, sin, cos, tan, pi
from numpy import array as _a

import vtkmodules.vtkInteractionStyle

from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleFlight,
    vtkInteractorStyleTerrain,
    vtkInteractorStyleUser
)

from utils import (
    dbg_print,
    vtkMatrix2array
)

_point_set_dtype_ = np.float32

def UpdatePropertyOTFScale(obj_prop, otf_s):
    pf = obj_prop.GetScalarOpacity()
    if hasattr(obj_prop, 'ref_prop'):
        obj_prop = obj_prop.ref_prop
    otf_v = obj_prop.prop_conf['opacity_transfer_function']['AddPoint']
    
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

    if hasattr(obj_prop, 'ref_prop'):
        obj_prop = obj_prop.ref_prop
    ctf_v = obj_prop.prop_conf['color_transfer_function']['AddRGBPoint']
    
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
    """ Guess values of colorscale for property otf and ctf in obj_prop. """
    otf_v, o_v = UpdatePropertyOTFScale(obj_prop, None)
    ctf_v, c_v = UpdatePropertyCTFScale(obj_prop, None)
    return o_v[-1][0] / otf_v[-1][0], c_v[-1][0] / ctf_v[-1][0]

def SetColorScale(obj_prop, scale):
    """ Set the color mapping for volume rendering. """
    dbg_print(4, 'Setting colorscale =', scale)
    if hasattr(scale, '__iter__'):
        otf_s = scale[0]
        ctf_s = scale[1]
    else:  # scalar
        otf_s = ctf_s = scale
    UpdatePropertyOTFScale(obj_prop, otf_s)
    UpdatePropertyCTFScale(obj_prop, ctf_s)

class execSmoothRotation():
    """ Continuously rotate camera. """
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
        dt = time_now - self.time_last_update
        self.actor.Azimuth(self.degree_per_sec * dt)
        self.time_last_update = time_now
        iren = obj
        iren.GetRenderWindow().Render()
        
        #ShotScreen(iren.GetRenderWindow(), \
        #    'pic_tmp/haha_t=%06.4f.png' % (time_now - self.time_start))
        #print('execSmoothRotation: Ren', time_now - self.time_start)

class RepeatingTimerHandler():
    """
    Repeatly execute `exec_obj` in a duration with fixed FPS.
    Requirements:
        exec_obj(obj, event, t_now)   Observer obj and event, parameter t_now.
        exec_obj.startat(t)           parameter t.
    Implimented by adding interactor observer TimerEvent.
    """
    def __init__(self, interactor, duration, exec_obj, fps = 30, b_fixed_clock_rate = False):
        self.exec_obj = exec_obj
        self.interactor = interactor
        self.timerId = None
        self.time_start = 0
        self.duration = duration
        self.fps = fps
        self.b_fixed_clock_rate = b_fixed_clock_rate

    def callback(self, obj, event):
        if self.b_fixed_clock_rate:
            self.tick += 1
            t_now = self.tick * 1/self.fps + self.time_start
        else:
            t_now = time.time()
        if t_now - self.time_start > self.duration:
            # align the time to the exact boundary
            t_now = self.time_start + self.duration
            self.exec_obj(obj, event, t_now)
            self.stop()
        else:
            self.exec_obj(obj, event, t_now)

    def start(self):
        self.ob_id = self.interactor.AddObserver('TimerEvent', self.callback)
        self.time_start = time.time()
        self.exec_obj.startat(self.time_start)
        self.timerId = self.interactor.CreateRepeatingTimer(int(1/self.fps))
        self.tick = 0
    
    def stop(self):
        if self.timerId:
            self.interactor.DestroyTimer(self.timerId)
            self.timerId = None
            self.interactor.RemoveObserver(self.ob_id)

    def __del__(self):
        self.stop()

class PointPicker():
    """
    Pick a point near the clicked site in the rendered scene.
    If multiple points exist, return only the nearest one.
    Input:
        points   : point set.
        renderer : scene renderer.
        posxy    : click site.
    Output:
        point ID, point coordinate
    """
    def __init__(self, points, renderer):
        ren_win = renderer.GetRenderWindow()
        cam = renderer.GetActiveCamera()
        self.GetViewParam(cam, ren_win.GetSize())
        self.p = np.array(points, dtype=np.float64)

    def GetViewParam(self, camera, screen_dims):
        # The matrix from cam to world
        # vec_cam = cam_m * vec_world
        # for cam_m =[[u v], inverse of it is:[[u.T  -u.T*v]
        #             [0 1]]                   [0     1    ]]
        self.cam_m = vtkMatrix2array(camera.GetModelViewTransformMatrix())
        self.screen_dims = _a(screen_dims)
        # https://vtk.org/doc/nightly/html/classvtkCamera.html#a2aec83f16c1c492fe87336a5018ad531
        view_angle = camera.GetViewAngle() / (180/pi)
        view_length = 2 * tan(view_angle/2)
        # aspect = width/height
        aspect_ratio = screen_dims[0] / screen_dims[1]
        if camera.GetUseHorizontalViewAngle():
            unit_view_window = _a([view_length, view_length/aspect_ratio])
        else:  # this is the default
            unit_view_window = _a([view_length*aspect_ratio, view_length])
        self.pixel_scale = unit_view_window / _a(screen_dims)

    def PickAt(self, posxy):
        cam_min_view_distance = 0
        selection_angle_tol = 0.01
        dbg_print(5, 'PickAt(): number of points:', self.p.shape[1])
        p = self.p.astype(_point_set_dtype_)
        # constructing picker line: r = v * t + o
        o = - self.cam_m[0:3,0:3].T @ self.cam_m[0:3, 3:4]  # cam pos in world
        o = o.astype(_point_set_dtype_)
        #   click pos in cam
        posxy_cam = (_a(posxy) - self.screen_dims / 2) * self.pixel_scale
        v = self.cam_m[0:3,0:3].T @ _a([[posxy_cam[0], posxy_cam[1], -1]]).T
        v = v.astype(_point_set_dtype_)
        # compute distance from p to the line r
        dbg_print(5, 'PickAt(): 2')
        u = p - o
        dbg_print(5, 'PickAt(): 3')
        t = (v.T @ u) / (v.T @ v)
        dbg_print(5, 'PickAt(): 4')
        dist = np.linalg.norm(u - v * t, axis=0)
        dbg_print(5, 'PickAt(): 6')
        angle_dist = dist / t
        dbg_print(5, 'PickAt(): 8')
        
        # find nearest point
        in_view_tol = (t > cam_min_view_distance) & (angle_dist < selection_angle_tol)
        ID_selected = np.flatnonzero(in_view_tol)
        if ID_selected.size > 0:
            angle_dist_selected = angle_dist[0, ID_selected]
            ID_selected = ID_selected[np.argmin(angle_dist_selected)]
        return ID_selected, p[:, ID_selected]

class PointSetHolder():
    def __init__(self):
        self._points_list = []
        self._len = 0
    
    def AddPoints(self, points, name):
        # TODO, maybe make it possible to find 'name' by point
        # points shape shoud be space_dim x index_dim
        self._points_list.append(points.astype(_point_set_dtype_))
        self._len += points.shape[1]
    
    def ConstructMergedArray(self):
        if len(self._points_list) > 1:
            a = np.concatenate(self._points_list, axis=1)
            self._points_list = [a]
            return a
        elif len(self._points_list) == 1:
            return self._points_list[0]
        else:
            return np.array([[],[],[]], dtype=_point_set_dtype_)
    
    def __len__(self):
        return self._len
    
    def __call__(self):
        return self.ConstructMergedArray()

class UIActions():
    """
    A collection of UI actions for key binding or mouse binding.
    As a rule of thumb, try not to use UI backend directly in this class.
    TODO: use the obj from the events.
      e.g.
        win = obj.iren.GetRenderWindow()
        rens = win.GetRenderers()
        rens.InitTraversal()
        ren1 = rens.GetNextItem()
        cam = ren1.GetActiveCamera()
    """
    def __init__(self, interactor, iren, guictrl):
        self.interactor = interactor
        self.iren = iren
        self.guictrl = guictrl

    def ExecByCmd(self, fn_name, get_attr_name = None):
        '''Call the action by name or list of name and arguments.'''
        dbg_print(4, "fn =", fn_name)
        if isinstance(fn_name, list):
            args = fn_name[1:]
            fn_name = fn_name[0]
        else:
            # fn_name should be a str, seperate arguments by spaces if any
            args = fn_name.split(' ')
            fn_name = args[0]
            args = args[1:]
        fn = getattr(self, fn_name.replace('-','_'))
        if get_attr_name:   # e.g. '__doc__'
            return getattr(fn, get_attr_name, None)
        fn(*args)

    def GetRenderers(self, n):
        '''currently it returns first two renderers'''
        rens = self.iren.GetRenderWindow().GetRenderers()
        rens.InitTraversal()
        ren1 = rens.GetNextItem()
        if n == 2:
            ren2 = rens.GetNextItem()
            return ren1, ren2
        elif n == 1:
            return ren1

    def auto_rotate(self):
        '''Animate rotate camera around the focal point.'''
        ren1, ren2 = self.GetRenderers(2)
        cam1 = ren1.GetActiveCamera()
        cam2 = ren2.GetActiveCamera()
        rotator = execSmoothRotation(cam1, 60.0)
        RepeatingTimerHandler(self.iren, 6.0, rotator, 100, True).start()

    def inc_brightness(self, cmd):
        '''Make the selected image darker or lighter.'''
        if not self.guictrl.selected_objects:
            return
        vol_name = self.guictrl.selected_objects[0]  # active object
        vol = self.guictrl.scene_objects[vol_name]
        obj_prop = vol.GetProperty()
        #obj_prop = self.guictrl.object_properties[vol_name]
        cs_o, cs_c = GetColorScale(obj_prop)
        k = sqrt(sqrt(2))
        if cmd.startswith('C'):
            k = sqrt(sqrt(k))
        if cmd.endswith('+'):
            k = 1.0 / k
        SetColorScale(obj_prop, [cs_o*k, cs_c*k])
        self.iren.GetRenderWindow().Render()         # TODO inform a refresh in a smart way
    
    def screen_shot(self):
        '''Save a screenshot to file.'''
        self.guictrl.ShotScreen()
    
    def save_scene(self):
        '''Save current scene to a project file.'''
        self.guictrl.ExportSceneFile()

    def fly_to_selected(self):
        '''Fly to selected object.'''
        if not self.guictrl.selected_objects:
            return
        vol_name = self.guictrl.selected_objects[0]  # active object
        dbg_print(4, 'Fly to:', vol_name)
        vol = self.guictrl.scene_objects[vol_name]
        bd = vol.GetBounds()
        center = [(bd[0]+bd[1])/2, (bd[2]+bd[3])/2, (bd[4]+bd[5])/2]
        ren1 = self.GetRenderers(1)
        self.iren.FlyTo(ren1, center)

    def fly_to_cursor(self):
        '''Fly to cursor.'''
        center = self.guictrl.Get3DCursor()
        if (center is not None) and (len(center) == 3):
            ren1 = self.GetRenderers(1)
            self.iren.FlyTo(ren1, center)
        else:
            dbg_print(3, 'No way to fly to.')

    def load_near_volume(self):
        '''load volume near cursor.'''
        center = self.guictrl.Get3DCursor()
        self.guictrl.LoadVolumeNear(center)
        self.iren.GetRenderWindow().Render()

    def set_view_up(self):
        '''Set camera view up right.'''
        dbg_print(4, 'Setting view up')
        ren1 = self.GetRenderers(1)
        cam1 = ren1.GetActiveCamera()
        cam1.SetViewUp(0,1,0)
        self.iren.GetRenderWindow().Render()

    def remove_selected_object(self):
        '''Remove the selected object.'''
        if len(self.guictrl.selected_objects) == 0:
            dbg_print(3, 'Nothing to remove.')
        else:
            obj_name = self.guictrl.selected_objects[0]
            self.guictrl.RemoveObject(obj_name)
            self.iren.GetRenderWindow().Render()

    def toggle_show_local_volume(self):
        '''Toggle showing of local volume.'''
        if self.guictrl.focusController.isOn:
            self.guictrl.focusController.Toggle()
        else:
            self.guictrl.focusController.Toggle()

    def exec_script(self):
        '''Run script.'''
        default_script_name = 'test_call.py'
        ren1 = self.GetRenderers(1)
        iren = self.iren
        print('Running script', 'test_call.py')
        try:
            exec(open(default_script_name).read())
            exec('PluginMain(ren1, iren, self.guictrl)')
        except Exception as inst:
            print('Failed to run due to exception:')
            print(type(inst))
            print(inst)

    def scene_zooming(self, direction, zooming_factor = 1.2):
        """Zoom in or out."""
        # direction ==  1: zoom in
        # direction == -1: zoom out
        ren1 = self.GetRenderers(1)
        cam = ren1.GetActiveCamera()
        # modify the distance between camera and the focus point
        fp = _a(cam.GetFocalPoint())
        p  = _a(cam.GetPosition())
        new_p = fp + (p - fp) * (zooming_factor ** (-direction))
        cam.SetPosition(new_p)
        # need to do ResetCameraClippingRange(), since VTK will
        # automatically reset clipping range after changing camera view
        # angle. Then the clipping range can be wrong for zooming.
        ren1.ResetCameraClippingRange()
        self.iren.GetRenderWindow().Render()

    def scene_object_traverse(self, direction):
        """Select next/previous scene object."""
        if self.guictrl.selected_pid:
            self.guictrl.SetSelectedPID(self.guictrl.selected_pid + direction)

    def camera_rotate_around(self):
        """Rotate the scene by mouse."""
        self.interactor.OnLeftButtonDown()   # vtkInteractorStyleTerrain
    
    def camera_rotate_around_release(self):
        self.interactor.OnLeftButtonUp()   # vtkInteractorStyleTerrain

    def camera_move_translational(self):
        """Move camera translationally in the scene."""
        self.interactor.OnMiddleButtonDown()   # vtkInteractorStyleTerrain

    def camera_move_translational_release(self):
        self.interactor.OnMiddleButtonUp()   # vtkInteractorStyleTerrain

    def select_a_point(self):
        """Select a point near the pointer."""
        ren = self.guictrl.GetMainRenderer()

        # select object
        # Ref. HighlightWithSilhouette
        # https://kitware.github.io/vtk-examples/site/Python/Picking/HighlightWithSilhouette/
        clickPos = self.iren.GetEventPosition()
        dbg_print(4, 'clicked at', clickPos)

        ppicker = PointPicker(self.guictrl.point_set_holder(), ren)
        pid, pxyz = ppicker.PickAt(clickPos)
        
        if pxyz.size > 0:
            dbg_print(4, 'picked point', pid, pxyz)
            self.guictrl.SetSelectedPID(pid)
        else:
            dbg_print(4, 'picked no point', pid, pxyz)
        # purposely no call to self.OnRightButtonDown()

def DefaultKeyBindings():
    """
    Full table of default key bindings. (except q for exit)
    See class UIAction for all available actions.
    Not that if there are multiple modifiers, i.e. Ctrl, Alt, Shift, they have to appear in
    the order Ctrl, Alt, Shift. and it is case sensitive.
    """
    d = {
        'r'        : 'auto-rotate',
        '+'        : 'inc-brightness +',
        'KP_Add'   : 'inc-brightness +',    # LEGION
        '-'        : 'inc-brightness -',
        'KP_Subtract': 'inc-brightness -',    # LEGION
        'Ctrl++'   : 'inc-brightness C+',
        'Ctrl+-'   : 'inc-brightness C-',
        'p'        : 'screen-shot',
        'Ctrl+s'   : 'save-scene',
        ' '        : 'fly-to-selected',
        '0'        : 'fly-to-cursor',
        'KP_0'     : 'fly-to-cursor',
        'KP_Insert': 'fly-to-cursor',         # LEGION
        'Return'   : 'load-near-volume',
        'KP_Enter' : 'load-near-volume',
        'KP_8'     : 'set-view-up',
        'KP_Up'    : 'set-view-up',           # LEGION
        'Shift+|'  : 'set-view-up',
        'Shift+\\' : 'set-view-up',           # LEGION
        'x'        : 'remove_selected_object',
        '`'        : 'toggle_show_local_volume',
        'Ctrl+g'   : 'exec-script',
        'MouseLeftButton'               : 'camera-rotate-around',
        'MouseLeftButtonRelease'        : 'camera-rotate-around-release',
        'Shift+MouseLeftButton'         : 'camera-move-translational',
        'Shift+MouseLeftButtonRelease'  : 'camera-move-translational-release',
        'MouseWheelForward'             : ['scene-zooming',  1],
        'MouseWheelBackward'            : ['scene-zooming', -1],
        'Shift+MouseWheelForward'       : ['scene-object-traverse',  1],
        'Shift+MouseWheelBackward'      : ['scene-object-traverse', -1],
        'MouseMiddleButton'             : 'camera-move-translational',
        'MouseMiddleButtonRelease'      : 'camera-move-translational-release',
        'MouseRightButton'              : 'select-a-point',
    }
    # For user provided key bindings we need to:
    # 1. Remove redundant white space.
    # 2. Sort order of the modifiers.
    # 3. Add release mappings to mouse button actions.
    return d

def GenerateKeyBindingDoc(key_binding, action):
    """Generate the key binding description from code, for help message."""
    s = "\n  Full key bindings:\n"
    for k, v in key_binding.items():
        h = action.ExecByCmd(v, get_attr_name = '__doc__')
        if h:
            l = 30 if 'Mouse' in k else 15
            s += ("%" + str(l) + "s : %s\n") % (k, h)
    return s

class MyInteractorStyle(vtkInteractorStyleTerrain):
    """
    Deal with keyboard and mouse interactions.

    Possible ancestor classes:
        vtkInteractorStyleTerrain
        vtkInteractorStyleFlight
        vtkInteractorStyleTrackballCamera
        vtkInteractorStyleUser
    """

    def __init__(self, iren, guictrl):
        self.iren = iren
        self.guictrl = guictrl

        # var for picker
        self.picked_actor = None

        # mouse events
        self.fn_modifier = []
        self.AddObserver('LeftButtonPressEvent',
                         self.left_button_press_event)
        self.AddObserver('LeftButtonReleaseEvent',
                         self.left_button_release_event)
        self.AddObserver('MiddleButtonPressEvent',
                         self.middle_button_press_event)
        self.AddObserver('MiddleButtonReleaseEvent',
                         self.middle_button_release_event)
        self.AddObserver('MouseWheelForwardEvent',
                         self.mouse_wheel_forward_event)
        self.AddObserver('MouseWheelBackwardEvent',
                         self.mouse_wheel_backward_event)
        self.AddObserver('RightButtonPressEvent',
                         self.right_button_press_event)
        self.AddObserver('RightButtonReleaseEvent',
                         self.right_button_release_event)

        for m in ['MouseLeftButton', 'MouseMiddleButton', 'MouseRightButton']:
            setattr(self, self.get_mb_var_name(m), '')

        # keyboard events
        self.AddObserver('CharEvent', self.OnChar)

        self.ui_action = UIActions(self, iren, guictrl)
        self.key_bindings = DefaultKeyBindings()

    def execute_key_cmd(self, key_combo, attr_name = None):
        if key_combo in self.key_bindings:
            fn_name = self.key_bindings[key_combo]
            return self.ui_action.ExecByCmd(fn_name, attr_name)
        return None

    def get_mb_var_name(self, mouse_button_name):
        return '_last_' + mouse_button_name + '_combo'

    def mouse_press_event_common(self, obj, mouse_button_name):
        modifier = self.get_key_modifier(obj.iren)
        cmd_st = modifier + mouse_button_name
        self.execute_key_cmd(cmd_st)
        setattr(self, self.get_mb_var_name(mouse_button_name), cmd_st)

    def mouse_release_event_common(self, obj, mouse_button_name):
        mb_var_name = self.get_mb_var_name(mouse_button_name)
        last_combo = getattr(self, mb_var_name, None)
        if last_combo:
            self.execute_key_cmd(last_combo + 'Release')
            setattr(self, mb_var_name, '')
        else:  # for unusual reason
            dbg_print(2, 'Singleton mouse button up.')
            self.OnLeftButtonUp()

    def left_button_press_event(self, obj, event):
        self.mouse_press_event_common(obj, 'MouseLeftButton')

    def left_button_release_event(self, obj, event):
        self.mouse_release_event_common(obj, 'MouseLeftButton')

    def mouse_wheel_forward_event(self, obj, event):
        modifier = self.get_key_modifier(obj.iren)
        self.execute_key_cmd(modifier + 'MouseWheelForward')

    def mouse_wheel_backward_event(self, obj, event):
        modifier = self.get_key_modifier(obj.iren)
        self.execute_key_cmd(modifier + 'MouseWheelBackward')

    def middle_button_press_event(self, obj, event):
        self.mouse_press_event_common(obj, 'MouseMiddleButton')
        return

    def middle_button_release_event(self, obj, event):
        self.mouse_release_event_common(obj, 'MouseMiddleButton')
        return

    def right_button_press_event(self, obj, event):
        self.mouse_press_event_common(obj, 'MouseRightButton')
    
    def right_button_release_event(self, obj, event):
        self.mouse_release_event_common(obj, 'MouseRightButton')
        return

    def get_key_modifier(self, iren):
        """Return key modifier, in fixed order (Ctrl, Alt, Shift)."""
        b_C = iren.GetControlKey()
        b_A = iren.GetAltKey()
        b_S = iren.GetShiftKey()  # sometimes reflected in key_code, like s and S
        key_modifier = ('Ctrl+'  if b_C else '') + \
                       ('Alt+'   if b_A else '') + \
                       ('Shift+' if b_S else '')
        return key_modifier

    def get_normalized_key_combo(self, iren):
        key_sym  = iren.GetKeySym()   # useful for PageUp etc.
        key_code = iren.GetKeyCode()
        key_modifier = self.get_key_modifier(iren)

        # normalize the key strike name
        if key_code < ' ':
            key_code = key_sym.replace('plus','+').replace('minus','-')
        key_combo = key_modifier + key_code
        #print('key_code:', bytearray(key_code.encode('utf-8')))
        dbg_print(4, 'Pressed:', key_combo, '  key_sym:', key_sym)

        # default key bindings in vtkInteractorStyleTerrain
        is_default_binding = (key_code.lower() in 'jtca3efprsuw') and \
                             ('Ctrl' not in key_modifier)
        # default key bindings interaction with control keys
        #        shift ctrl alt
        #    q    T    F    T
        #    3    F    F    T
        #    e    T    F    T
        #    r    T    F    T

        return key_combo, is_default_binding

    def OnChar(self, obj, event):
        """
        on keyboard stroke
        """
        iren = self.iren  # or obj.iren?
        
        key_combo, is_default_binding = self.get_normalized_key_combo(iren)

        self.execute_key_cmd(key_combo)

        # Leave all other key binding to the base vtk interactor.
        # Let's say, disable all default key bindings, except q.
        if not is_default_binding:
            super(MyInteractorStyle, obj).OnChar()
            # to quit, call TerminateApp()

