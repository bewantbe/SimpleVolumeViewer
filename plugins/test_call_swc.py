#

import numpy as np

class ArrayfyList:
    def __init__(self, obj_list):
        self.obj_list = obj_list
        self._build_index()

    def _build_index(self):
        self.obj_dict = {o.swc_name.split('#')[1]: o for o in self.obj_list}
    
    def __len__(self):
        return len(self.obj_list)
    
    def keys(self):
        return self.obj_dict.keys()

    def items(self):
        return self.obj_dict.items()

    def __getitem__(self, i):
        if isinstance(i, str):
            return self.obj_dict[i]
        else:
            return self.obj_list[i]

def PluginMain(ren, iren, gui_ctrl):
    swc_objs = gui_ctrl.GetObjectsByType('swc')
    print('Number of SWC:', len(swc_objs))

    s = ArrayfyList(swc_objs)

    for i in range(len(s)): s[i].visible = False
    s['487'].visible = True
    s['487'].color = (0.0, 1.0, 0)
    s['538'].visible = True
    s['538'].color = (1.0, 1.0, 0)

    print(np.min(s['538'].tree_swc[0], axis=0))
    print(np.min(s['538'].tree_swc[1], axis=0))

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
