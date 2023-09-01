# run in F2
# exec(open('/home/xyy/code/SimpleVolumeViewer/plugins/cmp_swc.py').read())
# for remembering the state

import os

if 'idx_show' not in locals():
    idx_show = 0
else:
    idx_show += 1

def PluginMain(ren, iren, gui_ctrl):
    swc_ids = [13, 30, 58, 96, 122, 155, 172, 192, 218, 251, 297, 373, 399, 419, 433, 446, 514, 530, 545, 554]
    version_list = ['v1.6.6', 'v1.7.0', 'v1.7.1', 'v1.7.2']

    if 1:
        objs = gui_ctrl.GetObjectsByType('swc')
        gui_ctrl.RemoveBatchObj(list(objs.keys()))

    k = idx_show

    root_dir = '/mnt/data_ext/swc_collect/RM009_manrefine/'

    path1 = f"{root_dir}{version_list[0]}/neuron#{swc_ids[k]}.lyp.swc"

    c_swc1 = {
        "type": "swc",
        "file_path": path1,
        "color": "Tomato",
        "line_width": 2.0,
    }

    # search for the last version
    for v in version_list[::-1]:
        path2 = f"{root_dir}{v}/neuron#{swc_ids[k]}.lyp.swc"
        if os.path.isfile(path2):
            break

    c_swc2 = {
        "type": "swc",
        "file_path": path2,
        "color": "Green",
        "line_width": 2.0,
    }

    gui_ctrl.AddObject('swc', c_swc1)
    gui_ctrl.AddObject('swc', c_swc2)

    ren.ResetCamera()

