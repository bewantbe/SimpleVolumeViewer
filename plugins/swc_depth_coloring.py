# Color the SWC by branch depth.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt  # must import for mpl.cm to work

from neu3dviewer.utils import (
    ArrayfyList,
    ArrayFunc,
    NamespaceOfSwcUtils,
)

# for remembering the state
if 'show_depth' not in locals():
    show_depth = False
show_depth ^= True       # switch the state

def PluginMain(ren, iren, gui_ctrl):
    user_ns = NamespaceOfSwcUtils(gui_ctrl, iren)
    swcs = user_ns['swcs']

    print('Number of SWC:', len(swcs))

    # get a color map (list of rgba)
    max_depth = 32          # number of colors for depth up-to
    n_depth_not_shown = 5   # n close to root, not to show
    if 0:
        f_cm = mpl.cm.get_cmap('viridis', max_depth - n_depth_not_shown)
        cm_table = f_cm.colors
    else:
        f_cm = mpl.cm.get_cmap('rainbow').reversed()
        cm_table = f_cm(np.linspace(0,1,max_depth - n_depth_not_shown))
    cm_table = np.vstack((np.zeros((n_depth_not_shown, 4)), cm_table))
    #cm_table[:,3] = 0.3

    # Convert color table to vtk LUT.
    lut = gui_ctrl.translator.prop_lut().parse({'lut':cm_table})

    # Color the SWC
    for s in swcs:
        if show_depth:
            s.ProcessColoring(max_depth = max_depth, lut = lut)
        else:
            s.ProcessColoring()

    # alter
    # ArrayFunc(lambda s: s.ProcessColoring())(swcs)
    # list(map(lambda s: s.ProcessColoring(), swcs))

    #for i in range(len(swcs)): swcs[i].visible = False
    #swcs.visible = False
    #swcs[['487', '538']].visible = True
    #swcs[['487', '538']].color = [(0.0, 1.0, 0), (1.0, 1.0, 0)]
    #swcs[:10].visible = True
    #swcs['487'].visible = True
    #swcs['487'].color = (0.0, 1.0, 0)
    #swcs['538'].visible = True
    #swcs['538'].color = (1.0, 1.0, 0)

#    print(np.min(swcs['538'].tree_swc[0], axis=0))
#    print(np.min(swcs['538'].tree_swc[1], axis=0))

#    swcs.visible = np.random.rand(len(swcs)) < 0.1

#    swcs.visible = True

#    for o in swc_objs:
#        o.color = np.random.rand(3)

#    for o in swc_objs:
#        idx = int(o.swc_name.split('#')[1])
#        if (idx == 450) or (idx == 396):
#            print('swc:', o.swc_name)
#            o.visible = True
#        else:
#            o.visible = False
