# SPDX-License-Identifier: GPL-3.0-or-later

# Keyboard and mouse interaction.
# All "actions" in the UI are here.

import time
import pprint
import json
import numpy as np
from numpy import sqrt, sin, cos, tan, pi
from numpy import array as _a
import numbers
import traceback

from vtkmodules.vtkCommonCore import (
    vtkCommand,
    vtkStringArray,
    VTK_STRING,
    VTK_OBJECT,
    VTK_INT,
)
from vtkmodules.util.misc import calldata_type
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleFlight,
    vtkInteractorStyleTerrain,
    vtkInteractorStyleUser
)
from vtkmodules.vtkRenderingCore import (
    vtkPropPicker,  # TODO: try them
    vtkPointPicker,
)
from vtkmodules.util.numpy_support import vtk_to_numpy

from .utils import (
    dbg_print,
    Struct,
    RotationMat,
    VecNorm,
    vtkMatrix2array,
    inject_swc_utils,
    IPython_embed,
    img_basic_stat,
)
from .data_loader import (
    TreeNodeInfo,
)

## Pollute the namespace with Events, prabably bad for IDE.
#_events = filter(lambda o: o[0].endswith('Event'), vars(vtkCommand).items())
#globals().update(dict(_events))

_point_set_dtype_ = np.float32

def DefaultKeyBindings():
    """GUI keyboard and mouse actions."""
    # See class UIAction for all available actions.
    # Not that if there are multiple modifiers, i.e. Ctrl, Alt, Shift, they have to appear in
    # the order Ctrl, Alt and Shift, and it is case sensitive.
    d = {
        'q'            : 'exit_program',
        'h'            : 'toggle_help_message',
        'p'            : 'screen_shot',
        'MouseLeftButton'               : 'camera_rotate_around',
        'MouseLeftButtonRelease'        : 'camera_rotate_around_release',
        'Shift+MouseLeftButton'         : 'camera_move_translational',
        'Shift+MouseLeftButtonRelease'  : 'camera_move_translational_release',
        'MouseMiddleButton'             : 'camera_move_translational',
        'MouseMiddleButtonRelease'      : 'camera_move_translational_release',
        'MouseWheelForward'             : ['scene_zooming',  1],
        'MouseWheelBackward'            : ['scene_zooming', -1],
        'MouseRightButton'              : 'select_a_point',
        'Ctrl+MouseRightButton'         : 'select_a_point append',
        'MouseLeftButtonDoubleClick'    : 'select_and_fly_to',
        'Shift+MouseWheelForward'       : ['scene_object_traverse',  1],
        'Shift+MouseWheelBackward'      : ['scene_object_traverse', -1],
        '0'            : 'fly_to_cursor',
        'KP_0'         : 'fly_to_cursor',
        'KP_Insert'    : 'fly_to_cursor',         # LEGION
        ' '            : 'fly_to_selected',
        'Shift+|'      : 'set_view_up',
        'Shift+\\'     : 'set_view_up',           # LEGION
        'Home'         : 'reset_camera_view',
        'r'            : 'auto_rotate',
        '+'            : 'inc_brightness +',
        'KP_Add'       : 'inc_brightness +',      # LEGION
        '-'            : 'inc_brightness -',
        'KP_Subtract'  : 'inc_brightness -',      # LEGION
        'Ctrl++'       : 'inc_brightness C+',
        'Ctrl+-'       : 'inc_brightness C-',
        'Return'       : 'load_near_volume',
        'KP_Enter'     : 'load_near_volume',
        'x'            : 'remove_selected_object',
        'w'            : ['select_near_3d_cursor', 100],
        '`'            : 'toggle_show_local_volume',
        'i'            : 'show_selected_info',
        'Ctrl+a'       : 'select_all_visible swc',
        'Ctrl+Shift+A' : 'deselect',
        'Ctrl+Shift+a' : 'deselect',
        'Ctrl+i'       : 'inverse_select swc',
        'Insert'       : 'toggle_hide_nonselected',
        'Alt+Return'   : 'toggle_fullscreen',
        'Ctrl+Return'  : 'toggle_stereo_mode',
        'Shift+Return' : 'toggle_stereo_mode next',
        'Ctrl+s'       : 'save_scene',
        'F2'           : 'embed_interactive_shell',
        'F5'           : 'refresh_plugin_key_bindings',
    }
    # For user provided key bindings we need to:
    # 1. Remove redundant white space.
    # 2. Sort order of the modifiers.
    # 3. Add release mappings to mouse button actions.
    return d

def QuickKeyBindingsHelpDoc():
    d = """
    Keyboard shortcuts:
        '+'/'-': Make the image darker or lighter;
                 Press also Ctrl to make it more tender;
        'r'    : Auto rotate the image for a while;
        'p'    : Take a screenshot and save it to TestScreenshot.png;
        ' '    : Fly to view the selected volume.
        '0'    : Fly to view the selected point in the fiber.
        'Enter': Load the image block (for Lychnis project).
        '|' or '8' in numpad: use Y as view up.
        Ctrl+s : Save the scene and view port.
        'q'    : Exit the program.

    Mouse function:
        left: drag to view in different angle;
        middle, left+shift: Move the view point.
        wheel: zoom;
        right click: select object, support swc points only currently.
    """
    return d

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
    Repeatedly execute `exec_obj` in a duration with fixed FPS.
    Requirements:
        exec_obj(obj, event, t_now)   Observer obj and event, parameter t_now.
        exec_obj.startat(t)           parameter t.
    Implemented by adding interactor observer TimerEvent.
    """
    def __init__(self, interactor, duration, exec_obj, fps = 30, b_fixed_clock_rate = False):
        self.exec_obj = exec_obj
        self.interactor = interactor
        self.timerId = None
        self.time_start = 0
        self.duration = duration
        self.fps = fps
        self.b_fixed_clock_rate = b_fixed_clock_rate

    @calldata_type(VTK_INT)
    def callback(self, obj, event, timer_id):
        if timer_id != self.timerId:
            return
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
        self.timerId = self.interactor.CreateRepeatingTimer(int(1000/self.fps))
        self.tick = 0
    
    def stop(self):
        if self.timerId:
            self.interactor.DestroyTimer(self.timerId)
            self.timerId = None
            self.interactor.RemoveObserver(self.ob_id)
            self.ob_id = None

    def __del__(self):
        self.stop()

class TimerHandler():
    """
    For quickly create a delayed action.
    
    Usually you need only one instance of `TimerHandler`,
    when needed you just insert the action by TimerHandler.schedule().
    
    Usage:
        timer_handler = TimerHandler()
        # interactor = vtkRenderWindowInteractor()
        timer_handler.Initialize(interactor)
        ...
        timer_handler.schedule(action, t_delay)
    """
    def __init__(self):
        self.interactor = None
        self._ob_id    = None
        # {timer_id: Struct(callback, finished, t_delay, t_inserted), ...}
        self.timer_callbacks = {}

    @calldata_type(VTK_INT)
    def callback(self, obj, event, timer_id):
        #assert event == 'TimerEvent'
        if timer_id not in self.timer_callbacks:
            return
        #dbg_print(5, 'TimerHandler::callback:', event, timer_id)
        timer = self.timer_callbacks[timer_id]
        timer.callback(obj)
        timer.finished = True
        del self.timer_callbacks[timer_id]

    def Initialize(self, iren):
        self.interactor = iren
        self._ob_id = self.interactor.AddObserver('TimerEvent', self.callback)

    def schedule(self, cb, t_delay):
        timer = Struct(callback=cb, finished = False, \
                       t_delay = t_delay, t_inserted = time.time())
        timer_id = self.interactor.CreateOneShotTimer(int(1000.0*t_delay))
        self.timer_callbacks[timer_id] = timer
        return timer

    def __del__(self):
        if self.interactor is None:
            return
        # destroy timers
        for i in self.timer_callbacks.keys():
            if self.interactor.IsOneShotTimer(i):
                self.interactor.DestroyTimer(i)
        del self.timer_callbacks
        # remove callback
        self.interactor.RemoveObserver(self._ob_id)

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
        self.InitViewParam(renderer)
        # if the same type avoid a copy
        self.p = points.astype(_point_set_dtype_, copy=False)

    def InitViewParam(self, renderer):
        ren_win = renderer.GetRenderWindow()
        win_size = ren_win.GetSize()
        self.in_stereo_mode = ren_win.GetStereoRender() and \
            ren_win.GetStereoTypeAsString() == 'SplitViewportHorizontal'
        
        camera = renderer.GetActiveCamera()

        # The matrix from cam to world
        # vec_cam = cam_m * vec_world
        # for cam_m =[[u v], inverse of it is:[[u.T  -u.T*v]
        #             [0 1]]                   [0     1    ]]
        self.cam_m = vtkMatrix2array(camera.GetModelViewTransformMatrix())
        self.win_size = _a(win_size)
        # https://vtk.org/doc/nightly/html/classvtkCamera.html#a2aec83f16c1c492fe87336a5018ad531
        view_angle = camera.GetViewAngle() / (180/pi)
        view_length = 2 * tan(view_angle/2)
        # aspect = width/height
        aspect_ratio = win_size[0] / win_size[1]
        if camera.GetUseHorizontalViewAngle():
            unit_view_window = _a([view_length, view_length/aspect_ratio])
        else:  # this is the default
            unit_view_window = _a([view_length*aspect_ratio, view_length])
        self.pixel_scale = unit_view_window / _a(win_size)

    def PickAt(self, posxy, ret_all = False):
        if self.in_stereo_mode:
            wx2 = self.win_size[0]/2
            dx = wx2 if posxy[0] > wx2 else 0
            posxy = (2*(posxy[0]-dx), posxy[1])
            dbg_print(4, 'PickAt()::stereo mode corrected point:', posxy)
    
        cam_min_view_distance = 0
        selection_angle_tol = 0.01
        dbg_print(5, 'PickAt(): number of points:', self.p.shape[1])
        # constructing picker line: r = v * t + o
        o = - self.cam_m[0:3,0:3].T @ self.cam_m[0:3, 3:4]  # cam pos in world
        o = o.astype(_point_set_dtype_, copy=False)
        #   click pos in cam
        posxy_cam = (_a(posxy) - self.win_size / 2) * self.pixel_scale
        v = self.cam_m[0:3,0:3].T @ _a([[posxy_cam[0], posxy_cam[1], -1]]).T
        v = v.astype(_point_set_dtype_, copy=False)
        # compute distance from p to the line r
        u = self.p - o
        t = (v.T @ u) / (v.T @ v)
        dist = VecNorm(u - v * t, axis=0)   # slow for large data set
        angle_dist = dist / t
        
        # find nearest point
        in_view_tol = (t > cam_min_view_distance) & (angle_dist < selection_angle_tol)
        ID_picked = np.flatnonzero(in_view_tol)
        angle_dist_picked = angle_dist[0, ID_picked]

        if ret_all:
            # return all candidates, sort by angle
            ind = np.argsort(angle_dist_picked)
            ID_picked = ID_picked[ind]
            # index(s) in point set, point position(s)
            return ID_picked, self.p[:, ID_picked]

        if ID_picked.size > 0:
            ID_picked = ID_picked[np.argmin(angle_dist_picked)]

        return ID_picked, self.p[:, ID_picked]

class PointSetHolder():
    def __init__(self):
        self._points_list = []
        self._len = 0
        self._point_set_boundaries = [0]
        self.name_list = []
        self._name_idx_map = {}
    
    def AddPoints(self, points, name):
        # points shape should be space_dim x index_dim
        self._points_list.append(points.astype(_point_set_dtype_, copy=False))
        self._len += points.shape[1]
        self._point_set_boundaries.append(self._len)
        self.name_list.append(name)
    
    def ConstructMergedArray(self):
        if len(self._points_list) > 1:
            a = np.concatenate(self._points_list, axis=1)
            self._points_list = [a]
            return a
        elif len(self._points_list) == 1:
            return self._points_list[0]
        else:
            return np.array([[],[],[]], dtype=_point_set_dtype_)
    
    def RemovePointsByName(self, name):
        # `name` can also be a name list
        if self._len == 0:
            return
        if isinstance(name, str):
            name = [name]
        name_idx_rm = [self.name_idx_map.get(nm, None) for nm in name]
        name_idx_rm = list(filter(lambda m: m is not None, name_idx_rm))
        name_idx_rm = np.unique(name_idx_rm)
        if len(name_idx_rm) == 0:
            return
        # Merge then remove
        a = self.ConstructMergedArray()
        psb = self._point_set_boundaries
        p_sieve = np.ones(a.shape[1], dtype=bool)
        for k in name_idx_rm:
            p_sieve[psb[k]:psb[k+1]] = False
        self._points_list[0] = a[:, p_sieve]
        # fix _point_set_boundaries
        nm_sieve = np.ones((len(self.name_list), ), dtype=bool)
        nm_sieve[name_idx_rm] = False
        blk_sz = np.diff(psb)
        blk_sz = blk_sz[nm_sieve]
        self._point_set_boundaries = [0] + list(np.cumsum(blk_sz))
        # fix _len
        self._len = self._point_set_boundaries[-1]
        # fix name_list
        self.name_list = [n for j, n in enumerate(self.name_list) \
                          if nm_sieve[j]]
    
    def GetSetidByPointId(self, point_id):
        set_id = np.searchsorted(self._point_set_boundaries,
                                 point_id, side='right') - 1
        return set_id

    def GetNameByPointId(self, point_id, unique = False):
        sid = self.GetSetidByPointId(point_id)
        if hasattr(point_id, '__len__'):
            if unique:
                return [self.name_list[i] for i in np.unique(sid)]
            else:
                return [self.name_list[i] for i in sid]
        else:
            return self.name_list[sid]
    
    def GetLocalPid(self, point_id):
        sid = self.GetSetidByPointId(point_id)
        if hasattr(point_id, '__len__'):
            return [point_id[j] - self._point_set_boundaries[i]
                    for j, i in enumerate(sid)]
        else:
            return point_id - self._point_set_boundaries[sid]
    
    def GetNameLocalPidByPointId(self, point_id):
        idx = self.GetSetidByPointId(point_id)
        name = self.name_list[idx]
        lid = point_id - self._point_set_boundaries[idx]
        return name, lid
    
    def __len__(self):
        return self._len
    
    def __call__(self):
        return self.ConstructMergedArray()

    @property
    def name_idx_map(self):
        if len(self._name_idx_map) != len(self.name_list):
            # build it!
            self._name_idx_map = {n:j for j, n in enumerate(self.name_list)}
        return self._name_idx_map

    def FindFirstObject(self, point_id_array, allowable_object_list):
        """
        Return the object name that corresponds to a point in point_id_array which:
        * the object is in allowable_object_list;
        * it is the first appeared object corresponds to points in point_id_array.
        """
        none = None, np.array([], dtype = point_id_array.dtype)
        if point_id_array.size == 0:
            return none

        if allowable_object_list is None:
            # select only the first object
            idx_choosen = 0
            pid = point_id_array[0]
        else:
            if len(allowable_object_list) == 0:
                return none
            name_idx_map = self.name_idx_map
            object_ids = [name_idx_map[n] for n in allowable_object_list]
            # naive algorithm, may be do it segment by segment?
            obj_idx_list = self.GetSetidByPointId(point_id_array)
            obj_hit = np.isin(obj_idx_list, object_ids)
            idx_hit = np.flatnonzero(obj_hit)
            dbg_print(4, 'FindFirstObject: found:', idx_hit)
            if len(idx_hit) == 0:
                return none
            idx_choosen = idx_hit[0]
            pid = point_id_array[idx_choosen]
        
        #obj_name = self.GetNameByPointId(pid)
        #return obj_name
        return pid, idx_choosen

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
    def __init__(self, interactor, iren, gui_ctrl):
        self.interactor = interactor
        self.iren = iren
        self.gui_ctrl = gui_ctrl

        # private variables for actions
        self.on_mouse_move_observer_id = {}
        self._rotation_assistant = Struct()

    def ExecByCmd(self, fn_name, get_attr_name = None):
        """Call the action by name or list of name and arguments."""
        if get_attr_name is None:
            dbg_print(4, "fn =", fn_name)
        if isinstance(fn_name, list):
            args = fn_name[1:]
            fn_name = fn_name[0]
        else:
            # fn_name should be a str, separate arguments by spaces if any
            args = fn_name.split(' ')
            fn_name = args[0]
            args = args[1:]
        fn = getattr(self, fn_name.replace('-','_'))
        if get_attr_name:   # e.g. '__doc__'
            return getattr(fn, get_attr_name, None)
        fn(*args)

    def GetRenderers(self, n):
        """currently it returns first two renderers"""
        rens = self.iren.GetRenderWindow().GetRenderers()
        rens.InitTraversal()
        ren1 = rens.GetNextItem()
        if n == 2:
            ren2 = rens.GetNextItem()
            return ren1, ren2
        elif n == 1:
            return ren1

    def exit_program(self):
        """Exit the program."""
        self.iren.TerminateApp()
        #self.ExitCallback()

    def auto_rotate(self):
        """Animate rotate camera around the focal point."""
        ren1, ren2 = self.GetRenderers(2)
        cam1 = ren1.GetActiveCamera()
        cam2 = ren2.GetActiveCamera()
        rotator = execSmoothRotation(cam1, 60.0)
        RepeatingTimerHandler(self.iren, 6.0, rotator, 60, True).start()

    def inc_brightness(self, cmd):
        """Make the selected image darker or lighter."""
        if not self.gui_ctrl.selected_objects or \
           len(self.gui_ctrl.selected_objects) == 0:
            return
        vol_name = self.gui_ctrl.selected_objects[0]  # active object
        k = sqrt(sqrt(2))
        if cmd.startswith('C'):
            k = sqrt(sqrt(k))
        if cmd.endswith('+'):
            k = 1.0 / k
        self.gui_ctrl.scene_objects[vol_name].set_color_scale_mul_by(k)
        self.gui_ctrl.Render()
    
    def auto_brightness(self, cmd):
        """Auto set selected volumn brightness."""
        if not self.gui_ctrl.selected_objects or \
           len(self.gui_ctrl.selected_objects) == 0:
            return
        vol_name = self.gui_ctrl.selected_objects[0]  # active object
        vol = self.gui_ctrl.scene_objects[vol_name]
        img3 = vol.actor.GetMapper().GetDataObjectInput()
        #dims = img3.GetDimensions()
        img_data = vtk_to_numpy(img3.GetPointData().GetScalars())
        stat = img_basic_stat(img_data)
        pprint.pprint(stat)
        #vol.modify()
    
    def screen_shot(self):
        """Save a screenshot in current directory."""
        self.gui_ctrl.ShotScreen()
    
    def save_scene(self):
        """Save current scene to a project file."""
        self.gui_ctrl.ExportSceneFile()

    def fly_to_selected(self):
        """Fly to selected object."""
        if not self.gui_ctrl.selected_objects:
            return
        vol_name = self.gui_ctrl.selected_objects[0]  # active object
        dbg_print(4, 'Fly to:', vol_name)
        center = self.gui_ctrl.scene_objects[vol_name].get_center()
        ren1 = self.GetRenderers(1)
        self.iren.FlyTo(ren1, center)

    def fly_to_cursor(self):
        """Fly to cursor."""
        center = self.gui_ctrl.Get3DCursor()
        if (center is not None) and (len(center) == 3):
            ren1 = self.GetRenderers(1)
            self.iren.FlyTo(ren1, center)
        else:
            dbg_print(3, 'No way to fly to.')

    def load_near_volume(self):
        """Load volume near cursor."""
        center = self.gui_ctrl.Get3DCursor()
        self.gui_ctrl.LoadVolumeNear(center)
        self.gui_ctrl.Render()

    def set_view_up(self):
        """Reset camera up direction to default."""
        dbg_print(4, 'Setting view up')
        ren1 = self.GetRenderers(1)
        cam1 = ren1.GetActiveCamera()
        cam1.SetViewUp(0,1,0)
        self.gui_ctrl.Render()

    def remove_selected_object(self):
        """Remove the selected object."""
        if len(self.gui_ctrl.selected_objects) == 0:
            dbg_print(3, 'Nothing to remove.')
        else:
            self.gui_ctrl.RemoveSelectedObjs()
            self.gui_ctrl.Render()

    def toggle_show_local_volume(self):
        """Toggle showing of local volume."""
        if self.gui_ctrl.focusController.isOn:
            self.gui_ctrl.focusController.Toggle()
        else:
            self.gui_ctrl.focusController.Toggle()

    def embed_interactive_shell(self):
        """Start an ipython session in command line with context, see %whos."""
        # Ref. IPython.terminal.embed.InteractiveShellEmbed
        # https://ipython.readthedocs.io/en/stable/interactive/reference.html#embedding-ipython
        # https://ipython.readthedocs.io/en/stable/api/generated/IPython.terminal.embed.html
        # old tutorial
        # https://ipython.org/ipython-doc/stable/interactive/reference.html#embedding-ipython
        # In the future, we may put this shell to an another thread, so that
        # the main UI is not freezed. For this to work we may need a customized
        # event(command/observer) to signal(notify) the main UI to read the
        # commands from the shell through a queue.
        #from IPython import embed
        self.gui_ctrl.StatusBar(' To return to this GUI, exit the interactive shell (CMD).')
        self.gui_ctrl.Render()
        from IPython.terminal.embed import InteractiveShellEmbed
        import IPython
        banner1 = "IPython interactive shell. " \
                  "Type 'exit' or press Ctrl+D to exit.\n"
        banner2 = inject_swc_utils.__doc__

        if not hasattr(self, '_shell_mode'):
            self._shell_mode = 'embed'

        if self._shell_mode == 'embed':
            # Simple usage:
            #     from IPython import embed
            #     embed()
            # See https://ipython.org/ipython-doc/stable/config/options/terminal.html
            # for possible parameters and configurations
            #
            # Note: IPython.embed has a limitation:
            # Inside the shell, we can't call complex list comprehensions.
            # Ref. https://stackoverflow.com/questions/35161324/how-to-make-imports-closures-work-from-ipythons-embed
            # such as:
            #     swc_objs = gui_ctrl.GetObjectsByType('swc')
            #     s = next(iter(swc_objs.values()))
            #     ty_s = type(s)
            #     prop_names = [ty_s for k in vars(s).keys()]
            # A workaround is to regist the variables first: \
            #    globals().update({'ty_s':ty_s})
            ns = locals()
            inject_swc_utils(ns, self)
            IPython_embed(colors="Neutral", # colors="Neutral" or "Linux"
                          header=banner2, confirm_exit = False)
        elif self._shell_mode == 'InteractiveShellEmbed':
            if not hasattr(self, 'embed_shell'):
                self.embed_shell = InteractiveShellEmbed(
                    banner1 = banner1,
                    banner2 = banner2,
                    confirm_exit = False)
                # auto reload module in IPython mode
                self.embed_shell.run_line_magic('reload_ext', 'autoreload')
                self.embed_shell.run_line_magic('autoreload', '2')

            ns = locals()
            inject_swc_utils(ns, self)
            # In new version of IPython (e.g. 8.11.0), direct use of 
            #   InteractiveShellEmbed could be broken:
            #   "Exception 'NoneType' object has no attribute 'check_complete'"
            #self.embed_shell(user_ns = ns)
            self.embed_shell(header = '', stack_depth=1, user_ns = ns)
        elif self._shell_mode == 'start_ipython':
            from IPython import start_ipython
            # we might try user_ns = locals() | globals()
            # start_ipython(argv=[], user_ns = locals())
            # The problem is, the second time we enter it, it will bug.
            # More precisely, after finishing the first start_ipython(),
            # the environemnt is already a mess.
            # Probably the only valid usage is to put start_ipython in a 
            # seperate thread.
            from traitlets.config import Config
            c = Config()
            c.InteractiveShellApp.exec_lines = [
                '%reload_ext autoreload',
                '%autoreload 2',
            ]
            ns = {}
            inject_swc_utils(ns, self)
            print(banner2)
            start_ipython( \
                argv=['--no-confirm-exit', '--no-banner'],
                config = c,
                user_ns = ns)
        else:
            dbg_print(1, "embed_interactive_shell: No such type of shell.")
        self.gui_ctrl.StatusBar(None)
        self.gui_ctrl.Render()

    def exec_script(self, script_name = 'test_call.py'):
        """Run a specific script."""
        ren1 = self.GetRenderers(1)
        iren = self.iren
        script_path = self.gui_ctrl.plugin_dir + script_name
        self.gui_ctrl.StatusBar(f' Running script: {script_name}')
        self.gui_ctrl.Render()
        dbg_print(3, 'Running script:', script_name)
        try:
            # running in globals() is a bit danger, any better idea?
            exec(open(script_path).read(), globals(), None)
            if 'PluginMain' in globals():
                ## noinspection PyUnresolvedReferences
                #PluginMain(ren1, iren, self.gui_ctrl)
                #exec('PluginMain(ren1, iren, self.gui_ctrl)')
                globals()['PluginMain'](ren1, iren, self.gui_ctrl)
                #locals()['PluginMain'](ren1, iren, self.gui_ctrl)
            else:
                dbg_print(2, 'Not defined "PluginMain(ren, iren, gui_ctrl)", assume script.')
        except Exception as inst:
            dbg_print(1, 'Failed to run due to exception:')
            dbg_print(1, type(inst))
            dbg_print(1, inst)
            traceback.print_exc()
        self.gui_ctrl.StatusBar(None)
        self.gui_ctrl.Render()

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
        self.gui_ctrl.Render()

    def scene_look_at(self, r_center, distance, direction = [1,1,1]):
        """Look at coordinate `r_center` at `distance`."""
        ren1 = self.GetRenderers(1)
        cam = ren1.GetActiveCamera()
        cam.SetFocalPoint(r_center)
        direction = _a(direction)
        new_p = r_center + direction / np.linalg.norm(direction) * distance
        cam.SetPosition(new_p)
        ren1.ResetCameraClippingRange()
        self.gui_ctrl.LazyRender()

    def scene_object_traverse(self, direction):
        """Select next/previous scene object, usually a point on swc."""
        if self.gui_ctrl.selected_pid:
            #self.gui_ctrl.SetSelectedPID(self.gui_ctrl.selected_pid + direction)
            self.mark_swc_point(self.gui_ctrl.selected_pid + direction)

    def camera_rotate_around(self):
        """Rotate view angle around the focus point."""
        iren = self.iren
        win = self.gui_ctrl.render_window
        ren = self.gui_ctrl.GetMainRenderer()
        cam = ren.GetActiveCamera()
        mouse_pos = _a(iren.GetEventPosition())
        # state of beginning of rotation
        ra = self._rotation_assistant
        ra.mouse_pos_init = mouse_pos
        ra.mouse_pos      = mouse_pos
        ra.window_size    = _a(win.GetSize())
        ra.window_pos     = _a(win.GetPosition())
        ra.mtime_init     = iren.GetMTime()
        ra.mtime          = ra.mtime_init
        # see PointPicker::InitViewParam on how to use this matrix
        ra.camera_mat_init = vtkMatrix2array(cam.GetModelViewTransformMatrix())
        ra.cam_focal_init  = _a(cam.GetFocalPoint())
        ra.cam_pos_init    = _a(cam.GetPosition())
        # pre computation
        r_rel_start = (ra.mouse_pos_init - ra.window_size / 2) / ra.window_size
        th_v0 = -pi*r_rel_start[1]
        th_h0 =  pi*r_rel_start[0]
        ra.rolling_space = RotationMat(th_v0, 'x') @ RotationMat(th_h0, 'y')
        # note down observer
        ob_id = self.interactor. \
            AddObserver(vtkCommand.MouseMoveEvent, self.camera_rotate_around_update)
        self.on_mouse_move_observer_id['rotate'] = ob_id
        # old implimentation
        #self.interactor.OnLeftButtonDown()    # vtkInteractorStyleTerrain
    
    def camera_rotate_around_update(self, e_obj, event):
        # Ref.
        # void vtkInteractorStyleTerrain::Rotate()
        # https://github.com/Kitware/VTK/blob/d706250a1422ae1e7ece0fa09a510186769a5fec/Interaction/Style/vtkInteractorStyleTerrain.cxx#L186
        iren = self.iren
        ren = self.gui_ctrl.GetMainRenderer()
        cam = ren.GetActiveCamera()
        mouse_pos = _a(iren.GetEventPosition())
        # compute rotation
        ra = self._rotation_assistant
        # relative positions, range [-0.5 ~ 0.5], 0 = middle of window
        r_rel       = (mouse_pos         - ra.mouse_pos_init ) / ra.window_size
        # rotation matrix
        th_vertical   = -pi*r_rel[1]
        #th_horizontal = pi*r_rel[0] * ra.window_size[0] / ra.window_size[1]
        th_horizontal = pi*r_rel[0]
        rot = RotationMat(th_vertical, 'x') @ RotationMat(th_horizontal, 'y')
        # denote cam_m =[[u, v],
        #                [0, 1]]
        # [vec_cam, 1]' = cam_m * [vec_world, 1]'
        # vec_cam = u * vec_world + v
        # vec_cam = u * (vec_world + u' * v)
        # invariants
        #   cam_focal
        #   u * (cam_pos - cam_focal) = u0 * (cam_pos_init - cam_focal)

        u0 = ra.camera_mat_init[0:3, 0:3]
        ## The old rotation is equivalent to 
        #u = RotationMat(th_vertical, 'x') @ u0 @ RotationMat(th_horizontal, 'y')
        u = ra.rolling_space @ rot @ ra.rolling_space.T @ u0
        cam_pos = u.T @ u0 @ (ra.cam_pos_init - ra.cam_focal_init) + ra.cam_focal_init
        v = u @ (-cam_pos)
        cam.SetRoll(0.0)
        cam.SetPosition(cam_pos)
        cam.SetFocalPoint(ra.cam_focal_init)
        cam.SetViewUp(u.T[:,1])

        #dbg_print(5, 'u =\n', u)
        #dbg_print(5, 'v =\n', v)

        ## the old rotation
        ## e_obj.Rotate()
        ## camera_mat = vtkMatrix2array(cam.GetModelViewTransformMatrix())
        ## dbg_print(5, 'rot =\n', camera_mat[0:3,0:3])

        iren.GetRenderWindow().Render()

    def camera_rotate_around_release(self):
        ob_id = self.on_mouse_move_observer_id.get('rotate', None)
        if ob_id is not None:
            self.interactor.RemoveObserver(ob_id)
        #self.interactor.OnLeftButtonUp()      # vtkInteractorStyleTerrain

    def camera_move_translational(self):
        """Move camera translationally in the scene."""
        self.interactor.OnMiddleButtonDown()  # vtkInteractorStyleTerrain

    def camera_move_translational_release(self):
        self.interactor.OnMiddleButtonUp()    # vtkInteractorStyleTerrain

    def select_a_point(self, select_mode = ''):
        """Select a point on fiber(SWC) near the click point."""
        ren = self.gui_ctrl.GetMainRenderer()

        clickPos = self.iren.GetEventPosition()
        dbg_print(4, 'Clicked at', clickPos)

        # selectable objects
        obj_swc = self.gui_ctrl.GetObjectsByType('swc')
        names_visible = [
            name for name, obj in obj_swc.items()
            if obj.visible == True
        ]
        pick_visible_mode = len(names_visible) != len(obj_swc)

        # "ray cone cast" select points
        points_holder = self.gui_ctrl.point_set_holder
        point_picker = PointPicker(points_holder(), ren)
        ppid, pxyz = point_picker.PickAt(clickPos, pick_visible_mode)

        if pick_visible_mode:
            # limit to selectable objects
            ppid, idx_choosen = points_holder.FindFirstObject(ppid, names_visible)
            dbg_print(4, 'PickAt(): Multi-pick: ppid =', ppid)
            pxyz = pxyz[:, idx_choosen]
        
        if pxyz.size == 0:
            ret = None
        else:
            obj_name, lid = points_holder.GetNameLocalPidByPointId(ppid)
            ret = (obj_name, lid, ppid, pxyz)
            swc_obj  = self.gui_ctrl.scene_objects[obj_name]
            assert np.sum(np.abs(pxyz - \
                     _point_set_dtype_(swc_obj.tree_swc[1][lid,:3]))) == 0
            if select_mode == 'append':
                if obj_name in self.gui_ctrl.selected_objects:
                    # remove
                    self.gui_ctrl.selected_objects.remove(obj_name)
                else:
                    # add
                    self.gui_ctrl.selected_objects.append(obj_name)
            dbg_print(4, 'Selected obj:', self.gui_ctrl.selected_objects)

        self.mark_swc_point(ret)
        self.measure_pick_distance(ret)
        return ret

    def mark_swc_point(self, pick_info):
        """
        Give information of the picked point and place the cursor on it.
        """
        # pick_info = (obj_name, lid, ppid, pxyz)
        if pick_info is None:
            dbg_print(4, 'PickAt(): picked no point.')
            self.gui_ctrl.InfoBar('')
            return

        if isinstance(pick_info, numbers.Integral):
            psh = self.gui_ctrl.point_set_holder
            if (pick_info >= psh().shape[1]) or (pick_info < 0):
                dbg_print(4, 'PickAt(): out-of-point set bound:', pick_info)
                return
            ppid = pick_info
            obj_name, lid = psh.GetNameLocalPidByPointId(ppid)
            swc_obj  = self.gui_ctrl.scene_objects[obj_name]
            pxyz = psh()[:, ppid]
        else:
            obj_name = pick_info[0]
            lid      = pick_info[1]
            ppid     = pick_info[2]
            pxyz     = pick_info[3]
        swc_obj  = self.gui_ctrl.scene_objects[obj_name]
        swc_name = swc_obj.swc_name
        swc_id   = swc_obj.tree_swc[0][lid, 0]
        l_info = TreeNodeInfo(swc_obj.tree_swc, lid)
        s_info = f'Picked: obj: "{obj_name}"\n' \
                 f' File: "{swc_name}"\n' \
                 f' SWC node: idx={lid}, id={swc_id}\n' \
                 f' xyz: {pxyz}\n' \
                 f' Branch depth: {l_info["branch_depth"]}\n' \
                 f' Node depth: {l_info["node_depth"]}\n' \
                 f' Path length to root: {l_info["root_distance"]:.1f}'
        dbg_print(4, s_info)
        #h = f'picked point: \nxyz = {pxyz} '
        #self.gui_ctrl.InfoBar({'type':'swc', 'obj_name':obj_name, 'header':h})
        self.gui_ctrl.InfoBar(s_info)
        self.gui_ctrl.SetSelectedPID(ppid)

    def measure_pick_distance(self, pick_info):
        """
        Measure distance between two recently picked points.

        pick_info = (obj_name, lid, ppid, pxyz)
        """
        if not hasattr(self, 'measure_point_queue'):
            self.measure_point_queue = []
            self.measure_point_queue_max_len = 10
        que = self.measure_point_queue
        if pick_info is None:
            que.clear()
            self.gui_ctrl.StatusBar(None)
            return
        # remove non-exist nodes
        while (len(que) > 0) and \
            (que[0][0] not in self.gui_ctrl.scene_objects):
                del que[0]
        while (len(que) > 1) and \
            (que[1][0] not in self.gui_ctrl.scene_objects):
                del que[1]
        # insert to queue and trim
        que.insert(0, pick_info)
        del que[self.measure_point_queue_max_len:]
        # give measurement
        if len(que) >= 2:
            dist = np.linalg.norm(que[0][3] - que[1][3])
            swc_obj0  = self.gui_ctrl.scene_objects[que[0][0]]
            swc_obj1  = self.gui_ctrl.scene_objects[que[1][0]]
            swc_name0 = swc_obj0.swc_name
            swc_name1 = swc_obj1.swc_name
            s = f'dist({swc_name0}:{que[0][1]}, {swc_name1}:{que[1][1]})' \
                f' = {dist:.1f}'
            self.gui_ctrl.StatusBar(s)

    def select_in_3d(self, r, radius, selection_mode = 'add'):
        """
        pick points and swcs in ball of center r and radius.
        r = np.array([53051.2, 27876.6, 52150.6])
        radius = 100
        """
        # get object names within range
        points_holder = self.gui_ctrl.point_set_holder
        points = points_holder().T
        id_sel = np.flatnonzero(np.linalg.norm(points - r, axis=1) <= radius)
        name_sel = points_holder.GetNameByPointId(id_sel, True)

        # ignore transparent objects
        obj_dict = self.gui_ctrl.scene_objects
        name_sel = [n for n in name_sel
                    if getattr(obj_dict[n], 'visible', False)]

        # adjust selection set
        so = self.gui_ctrl.selected_objects
        if selection_mode == 'add':
            so_new = set(so) | set(name_sel)
        elif selection_mode == 'subtract':
            so_new = set(so) - set(name_sel)
        elif selection_mode == 'xor':
            so_new = set(so) ^ set(name_sel)
        so.clear()
        so.extend(list(so_new))
        dbg_print(4, 'Selected obj:', self.gui_ctrl.selected_objects)

    def select_near_3d_cursor(self, radius):
        """Select SWCs near the cursor within a radius."""
        r = self.gui_ctrl.Get3DCursor()
        if r is None:
            return
        r = _a(r)
        self.select_in_3d(r, radius, 'add')
    
    def select_and_fly_to(self):
        """Pick the point at cursor and fly to it."""
        valid = self.select_a_point()
        if valid:
            self.fly_to_cursor()

    def deselect(self, select_mode = ''):
        """Deselect all selected objects."""
        # select_mode = all, reverse
        self.gui_ctrl.selected_objects.clear()
        dbg_print(4, 'selected obj:', self.gui_ctrl.selected_objects)

    def select_all_visible(self, obj_type):
        """Select all visible objects."""
        obj_swc = self.gui_ctrl.GetObjectsByType(obj_type, visible_only=True)
        self.gui_ctrl.selected_objects.clear()
        self.gui_ctrl.selected_objects.extend(obj_swc.keys())

    def inverse_select(self, obj_type):
        """Inverse select."""
        # full set
        obj_swc = self.gui_ctrl.GetObjectsByType(obj_type, visible_only=True)
        self.gui_ctrl.selected_objects = list(
                obj_swc.keys() - set(self.gui_ctrl.selected_objects))

    def toggle_help_message(self):
        """Toggle on screen key/mouse help."""
        self.gui_ctrl.ShowOnScreenHelp()

    def toggle_fullscreen(self):
        """Toggle full screen mode."""
        fs_state = self.gui_ctrl.win_conf.get('full_screen', False)
        dbg_print(4, 'Going to full screen mode', not fs_state)
        self.gui_ctrl.WindowConfUpdate(
            {'full_screen': not fs_state})

    def toggle_stereo_mode(self, toggle_mode = 'on-off'):
        """Toggle stereo(VR) mode. Optionally looping different stereo types."""
        win = self.iren.GetRenderWindow()
        # list of STEREO types:
        # https://vtk.org/doc/nightly/html/vtkRenderWindow_8h.html
        stereo_type_list = {
            1:'CrystalEyes',              2:'RedBlue',
            3:'Interlaced',               4:'Left',
            5:'Right',                    6:'Dresden',
            7:'Anaglyph',                 8:'Checkerboard',
            9:'SplitViewportHorizontal', 10:'Fake',
            11:'Emulate'}

        if 'stereo_type' not in self.gui_ctrl.win_conf:
            # Stereo mode is never been set
            # let's say, we want SplitViewportHorizontal to be the default
            stereo_type = 'SplitViewportHorizontal'
        else:
            # get current type from `win` so that we never loss sync
            # no matter win.GetStereoRender() is True or False, we can get type
            # There is a bug in GetStereoTypeAsString()
            #  https://github.com/collects/VTK/blob/master/Rendering/Core/vtkRenderWindow.cxx
            #  It do not have interlaced type and the Dresden type is called
            #  DresdenDisplay And more.
            stereo_type = stereo_type_list.get(win.GetStereoType(), '')
            if stereo_type == '':  # in wrong state.
                stereo_type = self.gui_ctrl.win_conf['stereo_type']
            if stereo_type == '':
                dbg_print(2, "toggle_stereo_mode(): What's wrong?")
                stereo_type = 'SplitViewportHorizontal'
        dbg_print(3, 'Current st: ', stereo_type)

        if toggle_mode == 'on-off':
            # on/off stereo
            if win.GetStereoRender():
                dbg_print(4, 'Stopping stereo mode')
                self.gui_ctrl.WindowConfUpdate(
                    {'stereo_type': ''})
            else:
                dbg_print(4, 'Going to stereo mode with type', stereo_type)
                self.gui_ctrl.WindowConfUpdate(
                    {'stereo_type': stereo_type})
        elif toggle_mode == 'next':
            # loop stereo types, let's say we don't want 'Fake' and 'Emulate'
            stereo_type_idx = list(stereo_type_list.values()).index(stereo_type)
            stereo_type_idx = (stereo_type_idx + 1) % (len(stereo_type_list) - 2)
            stereo_type = stereo_type_list[stereo_type_idx + 1]
            # enter stereo mode
            dbg_print(4, 'Going to stereo mode with type', stereo_type)
            self.gui_ctrl.WindowConfUpdate(
                {'stereo_type': stereo_type})
        else:
            dbg_print(4, 'toggle_stereo_mode() ???')
        win.Render()

    def toggle_hide_nonselected(self):
        """Toggle hiding non-selected object and showing all objects."""
        if not hasattr(self, 'hide_nonselected'):
            self.hide_nonselected = False
        # toggle
        self.hide_nonselected = not self.hide_nonselected
        # TODO: O(n^2) algorithm, should we optimize it?
        #  e.g. selected_objects use ordered set()
        #  or use set diff  scene_objects - selected_objects
        if self.hide_nonselected == True:
            dbg_print(5, 'Show only selected SWC.')
            for name, obj in self.gui_ctrl.scene_objects.items():
                if hasattr(obj, 'visible'):
                    if name in self.gui_ctrl.selected_objects:
                        obj.visible = True
                    else:
                        obj.visible = False
        else:
            dbg_print(5, 'Show all SWC.')
            for name, obj in self.gui_ctrl.scene_objects.items():
                if hasattr(obj, 'visible'):
                    obj.visible = True
        self.gui_ctrl.Render()

    def show_selected_info(self):
        """Show info about selected object(s)."""
        for name in self.gui_ctrl.selected_objects:
            obj_conf = self.gui_ctrl.scene_saved['objects'][name]
            print('Name:', name)
            pprint.pprint(obj_conf)

    def reset_camera_view(self):
        """Reset camera to view all objects."""
        self.gui_ctrl.GetMainRenderer().ResetCamera()
        #self.gui_ctrl.GetMainRenderer().ResetCameraClippingRange()
        self.gui_ctrl.Render()

    def refresh_plugin_key_bindings(self):
        """Refresh key bindings according to plugin/plugin.conf.json."""
        self.interactor.refresh_key_bindings()

def GenerateKeyBindingDoc(key_binding = DefaultKeyBindings(),
                          action = UIActions('', '', ''), help_mode = ''):
    """Generate the key binding description from code, for help message."""
    if help_mode == 'quick':
        s = QuickKeyBindingsHelpDoc()
        return s
    # Some function(command) binds to multiple keys, we better merge them.
    # get unique function items
    cmd_fn_list = map(lambda a:a[1], key_binding.items())
    cmd_name_fn_dict = {str(c):c for c in cmd_fn_list}
    # construct empty key list
    cmd_keys = {c:[] for c in cmd_name_fn_dict.keys()}
    # insert keys
    for k, v in key_binding.items():
        cmd_keys[str(v)].append(k)

    # key name refinement
    key_name_to_convention_map = {
        ' ': 'Space',     'Add': '+',       'Subtract': '-',
        'KP_': 'NumPad ', 'Return': 'Enter',
    }

    # generate help message
    left_margin = 4
    s = "\n  " + DefaultKeyBindings.__doc__ + "\n\n"
    for cmd_name, keys in cmd_keys.items():
        is_release = np.any([k.endswith('Release') for k in keys])
        if is_release:
            continue
        cmd = cmd_name_fn_dict[cmd_name]
        description = action.ExecByCmd(cmd, get_attr_name = '__doc__')
        # additional description on extra parameter(s)
        if isinstance(cmd, str) and cmd.count(' ') > 0:
            cmd_param = cmd.split(' ')[1:]
        elif isinstance(cmd, (list, tuple)) and len(cmd) > 1:
            cmd_param = str(cmd[1:])
        else:
            cmd_param = ''
        # convert key sym to conventional name
        for j, k in enumerate(keys):
            for old_key, new_key in key_name_to_convention_map.items():
                k = k.replace(old_key, new_key)
            keys[j] = k
        # different indent for mouse keys
        with_mouse = np.any(['Mouse' in k for k in keys])
        indent = (25 if with_mouse else 15) + left_margin
        # help lines
        line1 = f"{' ' * left_margin + ' or '.join(keys):>{indent}}"
        # newline if keystroke is long
        sep    = '\n' + ' ' * (indent) \
                 if (len(line1) > indent) else ''
        line2 = f"{sep} : {description} {cmd_param}\n"
        s += line1 + line2
    s += " \n"
    return s

class MouseDoubleClickHelper:
    max_click_interval = 0.33  # second
    max_click_dislocation = 5  # pixel
    
    def __init__(self):
        self.t_last = -1.0
        self.position = np.array([0, 0], dtype = np.int32)
        self.repeat_count = 0
    
    def ClickedAt(self, x, y):
        # typical usage:
        #   mdc.ClickedAt(x,y).repeat_count == 2
        t = time.time()
        if (t - self.t_last <= self.max_click_interval) and \
          (abs(x - self.position[0]) <= self.max_click_dislocation) and \
          (abs(y - self.position[1]) <= self.max_click_dislocation):
            self.repeat_count += 1
        else:
            self.repeat_count = 1
        self.position[0] = x
        self.position[1] = y
        self.t_last = t
        return self

    def Clicked(self, iren):
        clickPos = iren.GetEventPosition()
        return self.ClickedAt(clickPos[0], clickPos[1])

# TODO use vtkInteractorStyleUser at some point
class MyInteractorStyle(vtkInteractorStyleTerrain):
    """
    Deal with keyboard and mouse interactions.

    Possible ancestor classes:
        vtkInteractorStyleTerrain
        vtkInteractorStyleFlight
        vtkInteractorStyleTrackballCamera
        vtkInteractorStyleUser
    """
    # vtkWin32RenderWindowInteractor.cxx
    # https://github.com/Kitware/VTK/blob/d706250a1422ae1e7ece0fa09a510186769a5fec/Rendering/UI/vtkWin32RenderWindowInteractor.cxx
    # vtkInteractorStyleTerrain.cxx
    # https://github.com/Kitware/VTK/blob/d706250a1422ae1e7ece0fa09a510186769a5fec/Interaction/Style/vtkInteractorStyleTerrain.cxx

    user_event_cmd_id = vtkCommand.UserEvent + 1

    def __init__(self, iren, gui_ctrl):
        self.iren = iren
        self.gui_ctrl = gui_ctrl

        # var for picker
        self.picked_actor = None

        # A list of supported events:
        # https://vtk.org/doc/nightly/html/classvtkCommand.html

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
        # somehow VTK do not trigger LeftButtonDoubleClickEvent(at least v9.2)
        # we have to implement it ourself.
        #self.AddObserver(vtkCommand.LeftButtonDoubleClickEvent,
        #                 self.left_button_double_click_event)
        self.mdc = MouseDoubleClickHelper()

        for m in ['MouseLeftButton', 'MouseMiddleButton', 'MouseRightButton']:
            setattr(self, self.get_mb_var_name(m), '')

        # keyboard events
        # 'CharEvent' is not working on Windows for many of the keys and most of key combinations
        self.AddObserver('CharEvent', self.OnChar)
        self.AddObserver('KeyPressEvent', self.OnKeyPress)

        #self.gui_ctrl.GetMainRenderer(). \
        #    AddObserver(vtkCommand.EndEvent, self.OnRenderEnd)
        #self.t_last_update = time.time()
        #self.n_frames = 0

        self.AddObserver('DropFilesEvent', self.OnDropFiles)
        self.AddObserver(self.user_event_cmd_id, self.OnUserEventCmd)

        self.ui_action = UIActions(self, iren, gui_ctrl)
        self.key_bindings = DefaultKeyBindings()
        self.refresh_key_bindings()

    def refresh_key_bindings(self):
        """Refresh key bindings according to plugin/plugin.conf.json."""
        # try open the plugin.conf
        path = self.gui_ctrl.plugin_dir + '/plugin.conf.json'
        try:
            plugin_conf = json.loads(open(path).read())
        except Exception as inst:
            dbg_print(2, 'Failed to load plugins through path:', path)
            #dbg_print(1, inst)
            #traceback.print_exc()
            return
        shortcuts = plugin_conf['shortcuts']
        # insert the bindings
        self.key_bindings.update(shortcuts)
        dbg_print(3, 'Key bindings refreshed.')

    def bind_key_to_function(self, keystoke, uiact_func):
        if uiact_func is not None:
            setattr(UIActions, uiact_func.__name__, uiact_func)
            self.key_bindings[keystoke] = uiact_func.__name__
        else:
            if keystoke not in self.key_bindings:
                return
            name_fn = self.key_bindings[keystoke]
            if isinstance(name_fn, str):
                name_fn = [name_fn]
            name_fn = name_fn[0].split(' ')[0]
            if hasattr(UIActions, name_fn):
                delattr(UIActions, name_fn)
            self.key_bindings.pop(keystoke)

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
        if self.mdc.Clicked(self.iren).repeat_count == 2:  # double click
            self.mouse_press_event_common(obj, 'MouseLeftButtonDoubleClick')
        else:   # single click
            self.mouse_press_event_common(obj, 'MouseLeftButton')

#    def left_button_double_click_event(self, obj, event):
#        self.mouse_press_event_common(obj, 'MouseLeftButtonDoubleClick')

    def left_button_release_event(self, obj, event):
        if getattr(self, '_last_MouseLeftButtonDoubleClick_combo', None):
            setattr(self, '_last_MouseLeftButtonDoubleClick_combo', '')
        else:
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

    @staticmethod
    def check_is_default_binding(key_code, key_modifier):
        # default key bindings in vtkInteractorStyleTerrain
        # https://vtk.org/doc/nightly/html/classvtkInteractorStyle.html#details
        is_default_binding = (key_code.lower() in 'jtca3efprsuw') and \
                             ('Ctrl' not in key_modifier)
        # default key bindings interaction with control keys
        #        shift ctrl alt
        #    q    T    F    T
        #    3    F    F    T
        #    e    T    F    T
        #    r    T    F    T
        return is_default_binding

    def get_normalized_key_combo(self, iren):
        key_code = iren.GetKeyCode()
        key_sym  = iren.GetKeySym()   # useful for PageUp etc.
        key_modifier = self.get_key_modifier(iren)
        #print(f'key sym: {key_sym}, key_code: {key_code}, ord={ord(key_code)}')

        # normalize the key strike name
        if key_code < ' ':
            key_code = key_sym.replace('plus','+').replace('minus','-')
        if key_code in ['Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Shift_L', 'Shift_R']:
            key_code = ''  # they will exist in key_modifier (fix for windows)
            key_modifier = key_modifier[:-1]
        key_combo = key_modifier + key_code
        #print('key_code:', bytearray(key_code.encode('utf-8')))
        if key_combo not in ['Ctrl', 'Alt', 'Shift']:
            dbg_print(4, 'Pressed:', key_combo, '  key_sym:', key_sym)

        return key_combo

    def OnChar(self, obj, event):
        iren = obj.iren
        key_code = iren.GetKeyCode()
        #key_sym  = iren.GetKeySym()
        #dbg_print(5, f'OnChar(): key_code: "{key_code}", key_sym: "{key_sym}"')
        # Do not want 'default' key bindings.
        #super().OnChar()
        #super(MyInteractorStyle, obj).OnChar()
        if key_code == 'u':
            # Could InvokeEvent() be call by different thread?
            self.InvokeEvent(self.user_event_cmd_id)   

    def OnKeyPress(self, obj, event):
        """
        on keyboard stroke
        """
        iren = obj.iren
        key_combo = self.get_normalized_key_combo(iren)
        self.execute_key_cmd(key_combo)

        super().OnKeyPress()

    def OnRenderEnd(self, obj, event):
        self.n_frames += 1
        t = time.time()
        if t - self.t_last_update > 2.0:
            fps = self.n_frames / (t - self.t_last_update)
            dbg_print(5, f'FPS = {fps:.3f}')
            self.t_last_update = t
            self.n_frames = 0

    @calldata_type(VTK_OBJECT)
    def OnDropFiles(self, obj, event, calldata):
        # With help of:
        # VTK Python Wrappers - Observer Callbacks - Call Data
        # https://vtk.org/doc/nightly/html/md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html#observer-call-data
        # DropFilesEvent processing in vtkInteractorStyle::ProcessEvents
        # https://github.com/Kitware/VTK/blob/0fc87520d81ac51f48104bf5be7455eda3f365c3/Rendering/Core/vtkInteractorStyle.cxx#L1491
        # OnDropFiles Windows implementation
        # https://github.com/Kitware/VTK/blob/d706250a1422ae1e7ece0fa09a510186769a5fec/Rendering/UI/vtkWin32RenderWindowInteractor.cxx#L779
        assert isinstance(calldata, vtkStringArray)
        sa = calldata
        file_list = [sa.GetValue(j) for j in range(sa.GetNumberOfValues())]
        self.gui_ctrl.DropFilesObjectImporter(file_list)

    #@calldata_type(VTK_STRING)
    #def OnUserEventCmd(self, obj, event, calldata):
    def OnUserEventCmd(self, obj, event):
        # Ref.
        # https://kitware.github.io/vtk-examples/site/Cxx/Interaction/UserEvent/
        # https://vtk.org/doc/nightly/html/md__builds_gitlab_kitware_sciviz_ci_Documentation_Doxygen_PythonWrappers.html#observer-callbacks
        dbg_print(3, 'OnUserEventCmd() called.')
