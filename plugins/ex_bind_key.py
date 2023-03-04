# Plugin example.
# Create a custom function then dynamically bind it to a key.
# First press key to run this script, then press 'Ctrl+r' to execute.

# Alternative way to add key binding:
# Edit the file 'plugins/plugin.conf.json', then press F5 in the GUI make it in effect.

def custom_func(ui_act):
    ui_act.gui_ctrl.InfoBar('Rotating...')
    ui_act.reset_camera_view()
    ui_act.auto_rotate()
    def clear_info_bar(obj):
        ui_act.gui_ctrl.InfoBar('')
    ui_act.gui_ctrl.timer_handler.schedule(clear_info_bar, 3.0)

def PluginMain(ren, iren, gui_ctrl):
    gui_ctrl.BindKeyToFunction('Ctrl+r', custom_func)

    # remove it by:
    # gui_ctrl.BindKeyToFunction('Ctrl+r', None)
