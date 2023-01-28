#

import numpy as np
from utils import (
    ArrayfyList,
)

def PluginMain(ren, iren, gui_ctrl):
    swc_objs = gui_ctrl.GetObjectsByType('swc')
    print('Number of SWC:', len(swc_objs))

    s = ArrayfyList(swc_objs)
    print(len(s.color))

    #for i in range(len(s)): s[i].visible = False
    s.visible = False
    s[['487', '538']].visible = True
    s[['487', '538']].color = [(0.0, 1.0, 0), (1.0, 1.0, 0)]
    s[:10].visible = True
    #s['487'].visible = True
    #s['487'].color = (0.0, 1.0, 0)
    #s['538'].visible = True
    #s['538'].color = (1.0, 1.0, 0)

#    print(np.min(s['538'].tree_swc[0], axis=0))
#    print(np.min(s['538'].tree_swc[1], axis=0))

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

    iren.GetRenderWindow().Render()
    # TODO: make it possible to open a ipython environment, like Matlab.
