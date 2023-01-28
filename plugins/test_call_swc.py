#

import numpy as np

def bind_property_broadcasting(li, pn, docs):
    # With help of:
    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    # https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python
    
    '''
    def bind(instance, func, as_name=None):
        """
        Bind the function *func* to *instance*, with either provided name *as_name*
        or the existing name of *func*. The provided *func* should accept the 
        instance as the first argument, i.e. "self".
        """
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(instance, instance.__class__)
        setattr(instance, as_name, bound_method)
        return bound_method
    '''

    def getx(li):
        return [getattr(o, pn) for o in li]
    #getx_bind = getx.__get__(li, li.__class__)
    #setattr(li, 'get_'+pn, getx_bind)
    
    def setx(li, x):
        if isinstance(x, (list, np.ndarray, ArrayfyList)):
            # array assignment
            if len(x) != len(li):
                raise ValueError(f'Length not match: assign length {len(x)} to length {len(li)}.')
            for j, o in enumerate(li):
                setattr(o, pn, x[j])
        else:
            # scalar assignment
            for o in li:
                setattr(o, pn, x)
    #setx_bind = setx.__get__(li, li.__class__)
    #setattr(li, 'set_'+pn, setx_bind)

    #setattr(type(li), pn, property(getx_bind, setx_bind, None, docs))
    setattr(type(li), pn, property(getx, setx, None, docs))

class ArrayfyList:
    """Wrapper for list of swc objects. Support for other objects is possible."""
    def __init__(self, obj_list):
        self.obj_list = obj_list
        setattr(self, '__iter__', obj_list.__iter__)
        self.rebuild_index()
        self._bind_properties()

    def rebuild_index(self, index_style = 'numeric'):
        if index_style == 'numeric':
            self.obj_dict = {o.swc_name.split('#')[1]: o for o in self.obj_list}
        elif index_style == 'file_name':
            self.obj_dict = {o.swc_name: o for o in self.obj_list}
        else:
            raise TypeError(f'Unknown index style "{index_style}".')
    
    def _bind_properties(self):
        if len(self.obj_list) == 0:
            return
        s = self.obj_list[0]
        ty_s = type(s)
        # get property attributes, in the form _prop
        prop_names = [k[1:] for k in vars(s).keys() \
                           if k.startswith('_') and \
                              isinstance(getattr(ty_s, k[1:], None), property)]
        for pn in prop_names:
            bind_property_broadcasting(self, pn, getattr(ty_s, pn).__doc__)

    def __len__(self):
        return len(self.obj_list)
    
    def keys(self):
        return self.obj_dict.keys()

    def items(self):
        return self.obj_dict.items()

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # index by swc name
            return self.obj_dict[idx]
        elif isinstance(idx, (slice, range)):
            idx_max = self.__len__()
            idx_range = [0, idx_max, 1]     # default values
            if idx.start is not None:
                idx_range[0] = idx.start % idx_max
            if idx.stop is not None:
                idx_range[1] = idx.stop  % idx_max
            if idx.step is not None:
                idx_range[2] = idx.step
            return ArrayfyList([self.__getitem__(i) for i in range(*idx_range)])
        elif isinstance(idx, (list, np.ndarray)):
            # array style index
            return ArrayfyList([self.__getitem__(i) for i in idx])
        else:
            # index by order in the list
            return self.obj_list[idx]

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
