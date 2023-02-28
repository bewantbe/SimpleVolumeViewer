#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from neu3dviewer.utils import (
    ArrayfyList,
    ArrayFunc,
    NamespaceOfSwcUtils,
)
from neu3dviewer.data_loader import (
    SplitSWCTree,
    SimplifyTreeWithDepth,
)

def PluginMain(ren, iren, gui_ctrl):
    user_ns = NamespaceOfSwcUtils(gui_ctrl, iren)
    swcs = user_ns['swcs']

    print('Number of SWC:', len(swcs))

    # swc depth coloring
    max_depth = 32

    n_depth_not_shown = 2
    cm_table = mpl.cm.get_cmap('viridis', max_depth - n_depth_not_shown).colors
    cm_table = np.vstack((np.zeros((n_depth_not_shown, 4)), cm_table))
    lut = gui_ctrl.translator.prop_lut().parse({'lut':cm_table})

    for s in swcs:
        #print('Processing', s.swc_name)
        s.ProcessColoring(max_depth = max_depth, lut = lut)
        #s.ProcessColoring()

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

#    for o in swc_objs:
#        o.visible = np.random.rand() < 0.1

#    for o in swc_objs:
#        o.visible = True

#    for o in swc_objs:
#        o.color = np.random.rand(3)

#    for o in swc_objs:
#        idx = int(o.swc_name.split('#')[1])
#        if (idx == 450) or (idx == 396):
#            print('swc:', o.swc_name)
#            o.visible = True
#        else:
#            o.visible = False
