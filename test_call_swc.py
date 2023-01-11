#

import numpy as np

def PluginMain(ren, iren, guictrl):
    swc_objs = guictrl.GetObjectsByType('swc')
    print('Number of SWC:', len(swc_objs))
#    for o in swc_objs:
#        o.visible = np.random.rand() < 0.1
    for o in swc_objs:
        o.color = np.random.rand(3)
    iren.GetRenderWindow().Render()
    # TODO: make it possible to open a ipython environment, like Matlab.
