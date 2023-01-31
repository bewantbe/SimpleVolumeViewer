#

import numpy as np
from neu3dviewer.utils import (
    ArrayfyList,
    ArrayFunc,
    NamespaceOfSwcUtils,
)

def PluginMain(ren, iren, gui_ctrl):
    user_ns = NamespaceOfSwcUtils(gui_ctrl, iren)
    swcs = user_ns['swcs']

    print('Number of SWC:', len(swcs))

    #for i in range(len(swcs)): swcs[i].visible = False
    swcs.visible = False
    swcs[['487', '538']].visible = True
    swcs[['487', '538']].color = [(0.0, 1.0, 0), (1.0, 1.0, 0)]
    swcs[:10].visible = True
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
