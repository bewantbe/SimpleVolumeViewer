# SPDX-License-Identifier: GPL-3.0-or-later

# Small utility functions

import datetime
import numpy as np

from vtkmodules.vtkCommonDataModel import (
    vtkColor3ub,
    vtkColor3d,
    vtkColor4d,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
)

debug_level = 5

def dbg_print(level, *p, **keys):
    """
    Used for printing error and debugging information.
    Controlled by global (module) debug_level.
    Higher debug_level will show more information.
        debug_level == 0: show nothing.
        debug_level == 1: show only error.
        debug_level == 2: show warning.
        debug_level == 3: show hint.
        debug_level == 4: show message.
        debug_level == 5: most verbose.
    """
    if level > debug_level:
        return
    level_str = {1:'Error', 2:'Warning', 3:'Hint', 4:'Message', 5:'Verbose'}
    print(level_str[level] + ':', *p, **keys)

def str2array(s, sep = ' ', dtype=float):
    """ Convert list of numbers in string form to `list`. """
    if not isinstance(s, str):
        return s
    if s[0] == '[':
        # like '[1,2,3]'
        v = [dtype(it) for it in s[1:-1].split(',')]
    else:
        # like '1 2 3', '1024x768' (sep='x')
        v = [dtype(it) for it in s.split(sep)]
    return v

def _mat3d(d):
    """ Convert vector of length 9 to 3x3 numpy array. """
    return np.array(d, dtype=np.float64).reshape(3,3)

def array2str(a, prec = 4, sep = '\n'):
    return sep + np.array_str(np.array(a), precision = prec, suppress_small = True)

def vtkGetColorAny3d(c):
    if isinstance(c, str):
        colors = vtkNamedColors()
        return colors.GetColor3d(c)
    elif isinstance(c, vtkColor3ub):
        return vtkColor3d(c[0]/255, c[1]/255, c[2]/255)
    elif isinstance(c, vtkColor3d):
        return c
    elif hasattr(c, '__len__') and (len(c) == 3 or len(c) == 4):
        # tuple, list, numpy array of 3 floating numbers
        # ignore alpha channel, if you want, try vtkGetColorAny4d
        if (c[0]>1.0) or (c[1]>1.0) or (c[2]>1.0):
            dbg_print(3, 'vtkGetColorAny(): assuming uint8*3 color.')
            return vtkColor3d(c[0]/255, c[1]/255, c[2]/255)
        return vtkColor3d(c[0], c[1], c[2])
    else:
        return c

def vtkGetColorAny4d(c):
    if isinstance(c, str):
        colors = vtkNamedColors()
        return colors.GetColor4d(c)
    elif hasattr(c, '__len__') and (len(c) == 3 or len(c) == 4):
        # tuple, list, numpy array of 3 or 4 floating numbers
        a = c[3] if len(c) == 4 else 1.0
        return vtkColor4d(c[0], c[1], c[2], a)
    else:
        return c

def vtkMatrix2array(vtkm):
    """ Convert VTK matrix to numpy matrix (array) """
    # also use self.cam_m.GetData()[i+4*j]?
    m = np.array(
            [
                [vtkm.GetElement(i,j) for j in range(4)]
                for i in range(4)
            ], dtype=np.float64)
    return m

def rg_part_to_pixel(rg, max_pixel):
    """
    Utilizer to convert a fraction to integer range.
    Mostly copy from VISoR_select_light/pick_big_block/volumeio.py
    Examples:
      rg=[(1, 2)], max_pixel=100: return ( 0,  50)
      rg=[(2, 2)], max_pixel=100: return (50, 100)
      rg=[],       max_pixel=100: return ( 0, 100)
      rg=(0, 50),  max_pixel=100: return ( 0,  50)
      rg=([0.1], [0.2]), max_pixel=100: return ( 10,  20)
    """
    if len(rg) == 0:
        return (0, max_pixel)
    elif len(rg)==1 and len(rg[0])==2:
        # in the form rg=[(1, 2)], it means 1/2 part of a range
        rg = rg[0]
        erg = (int((rg[0]-1)/rg[1] * max_pixel), 
               int((rg[0]  )/rg[1] * max_pixel))
        return erg
    elif len(rg)==2 and isinstance(rg[0], (list, tuple)):
        # in the form rg=([0.1], [0.2]), means 0.1~0.2 part of a range
        p0, p1 = rg[0][0], rg[1][0]
        erg = [int(p0 * max_pixel), int(p1 * max_pixel)]
        return erg
    else:  # return as-is
        return rg

def slice_from_str(slice_str):
    """
    Construct array slice object from string.
    Ref: https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
    Example:
        slice_str = "[100:400, :, 20:]"
        return: (slice(100,400), slice(None,None), slice(20,None))
    """
    dim_ranges = slice_str[1:-1].split(',')
    # convert a:b:c to slice(a,b,c)
    dim_ranges = tuple(
                     slice(
                         *map(
                             lambda x: int(x.strip())
                                 if x.strip() else None,
                             rg.split(':')
                         ))
                     for rg in dim_ranges
                 )
    return dim_ranges

def GetNonconflitName(prefix, name_set):
    """ Return a name start with prefix but not occurred in name_set. """
    i = 1
    name = prefix
    while name in name_set:
        name = prefix + '.%.3d'%i
        i += 1
    return name

def MergeFullDict(d_contain, d_update):
    """
    Update dict d_contain by d_update.
    It is a "deep update" version of the dict.update().
    i.e. overwrite d_contain for items exist in d_update
    Ref. https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-take-union-of-dictionari
    """
    def DeepUpdate(d_contain, d_update):
        for key, value in d_update.items(): 
            if key not in d_contain:
                d_contain[key] = value
            else:  # key in d_contain
                if isinstance(value, dict):
                    DeepUpdate(d_contain[key], value)
                else:  # overwrite
                    # simple sanity check: data type must agree
                    if type(d_contain[key]) == type(value):
                        d_contain[key] = value
                    else:
                        dbg_print(2, 'DeepUpdate()', 'key type mismatch! value discard.')
        return d_contain

    DeepUpdate(d_contain, d_update)

    return d_contain

def UpdatePropertyOTFScale(obj_prop, otf_s):
    pf = obj_prop.GetScalarOpacity()
    if hasattr(obj_prop, 'ref_prop'):
        obj_prop = obj_prop.ref_prop
    otf_v = obj_prop.prop_conf['opacity_transfer_function']['AddPoint']
    
    # initialize an array of array
    # get all control point coordinates
    v = np.zeros((pf.GetSize(), 4))
    for k in range(pf.GetSize()):
        pf.GetNodeValue(k, v[k])

    if otf_s is None:  # return old otf and current setting
        return otf_v, v

    for k in range(pf.GetSize()):
        v[k][0] = otf_s * otf_v[k][0]
        pf.SetNodeValue(k, v[k])

def UpdatePropertyCTFScale(obj_prop, ctf_s):
    ctf = obj_prop.GetRGBTransferFunction()
    # get all control point coordinates
    # location (X), R, G, and B values, midpoint (0.5), and sharpness(0) values

    if hasattr(obj_prop, 'ref_prop'):
        obj_prop = obj_prop.ref_prop
    ctf_v = obj_prop.prop_conf['color_transfer_function']['AddRGBPoint']
    
    # initialize an array of array
    # get all control point coordinates
    v = np.zeros((ctf.GetSize(), 6))
    for k in range(ctf.GetSize()):
        ctf.GetNodeValue(k, v[k])

    if ctf_s is None:  # return old ctf and current setting
        return ctf_v, v

    for k in range(ctf.GetSize()):
        v[k][0] = ctf_s * ctf_v[k][0]
        ctf.SetNodeValue(k, v[k])

def GetColorScale(obj_prop):
    """ Guess values of colorscale for property otf and ctf in obj_prop. """
    otf_v, o_v = UpdatePropertyOTFScale(obj_prop, None)
    ctf_v, c_v = UpdatePropertyCTFScale(obj_prop, None)
    return o_v[-1][0] / otf_v[-1][0], c_v[-1][0] / ctf_v[-1][0]

def SetColorScale(obj_prop, scale):
    """ Set the color mapping for volume rendering. """
    dbg_print(4, 'Setting colorscale =', scale)
    if hasattr(scale, '__iter__'):
        otf_s = scale[0]
        ctf_s = scale[1]
    else:  # scalar
        otf_s = ctf_s = scale
    UpdatePropertyOTFScale(obj_prop, otf_s)
    UpdatePropertyCTFScale(obj_prop, ctf_s)

def ConditionalAddItem(name, cmd_obj_desc, key_name, win_conf):
    if key_name == '':
        key_name = name
    if name in cmd_obj_desc:
        win_conf.update({
            key_name: cmd_obj_desc[name]
        })

def WindowsFriendlyDateTime():
    # return something like '2023-01-18_01h10m59.65'
    st_time = str(datetime.datetime.now()).replace(' ', '_') \
                    .replace(':', 'h', 1).replace(':', 'm', 1)[:22]
    return st_time

def GetRangeTuple(idx, idx_max):
    """Input: slice or range, output: numerical [start, end, step]"""
    # usage example:
    # idx_range = GetRangeTuple(idx, self.__len__())
    # for j, i in enumerate(range(*idx_range)):
    idx_range = [0, idx_max, 1]     # default values
    if idx.start is not None:
        idx_range[0] = idx.start % idx_max
    if idx.stop is not None:
        idx_range[1] = idx.stop  % idx_max
    if idx.step is not None:
        idx_range[2] = idx.step
    return idx_range

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
    """
    Wrapper for list of SWC objects, so that has a numpy array style access.
    Essentially automatically broadcast the properties of the items.
    Support for objects other than SWC is possible.
    """
    def __init__(self, obj_list):
        if not isinstance(obj_list, (list, np.ndarray, tuple)):
            raise TypeError('Wrapper for list only.')
        self.obj_list = obj_list
        setattr(self, '__iter__', obj_list.__iter__)
        #setattr(self, '__len__',  obj_list.__len__)  # need to rebind
        self.rebuild_index()
        self._bind_properties()

    def list(self):
        return self.obj_list

    def rebuild_index(self, index_style = 'numeric'):
        """
        Rebuild indexing string according style.
        index_style can be 'numeric'
        """
        # guess the type
        if self.__len__() == 0:
            self.obj_dict = {}
            return
        
        s = self.obj_list[0]
        if hasattr(s, 'swc_name'):  # should be a swc file
            if index_style == 'numeric':
                fn = lambda j, o: o.swc_name.split('#')[1]
            elif index_style == 'file_name':
                fn = lambda j, o: o.swc_name
            else:
                raise TypeError(f'Unknown index style "{index_style}".')
        else:
            fn = lambda j, o: str(j)

        self.obj_dict = {fn(j, o): o for j, o in enumerate(self.obj_list)}
    
    def _bind_properties(self):
        if len(self.obj_list) == 0:
            return
        s = self.obj_list[0]
        if not hasattr(s, '__dict__'):
            return
        ty_s = type(s)
        # get property attributes, in the form _prop
        prop_names = [k[1:] for k in vars(s).keys() \
                           if k.startswith('_') and \
                              isinstance(getattr(ty_s, k[1:], None), property)]
        for pn in prop_names:
            bind_property_broadcasting(self, pn, getattr(ty_s, pn).__doc__)

    def __len__(self):
        return self.obj_list.__len__()

    def keys(self):
        return self.obj_dict.keys()

    def items(self):
        return self.obj_dict.items()

    def __repr__(self):
        return self.obj_list.__repr__()

    def __str__(self):
        s = '[' + ', '.join([f'"{k}"' for k in self.obj_dict.keys()]) + ']'
        return s

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # index by swc name, like "['123']"
            return self.obj_dict[idx]
        elif isinstance(idx, slice):
            # indexing by slice or range, like: "[:10]"
            return ArrayfyList(self.obj_list[idx])
        elif isinstance(idx, (list, np.ndarray)):
            # array style index, like "[["1", "2", "3"]]"
            return ArrayfyList([self.__getitem__(i) for i in idx])
        else:
            # index by order in the list, like "[123]"
            return self.obj_list[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, str):
            raise TypeError('Assignment by string is not allowed.')
        elif isinstance(idx, slice):
            # indexing by slice or range, like: "[:10]"
            self.obj_list[idx] = val
        elif isinstance(idx, (list, np.ndarray)):
            # array style index, like "[[1, 2, 3]]"
            for j, i in enumerate(idx):
                self.obj_list[i] = val[j]
        else:
            # index by order in the list, like "[123]"
            self.obj_list[idx] = val
        self.rebuild_index()

def ArrayFunc(func):
    """
    Arrayfy the func such that it accept array(list) input, i.e. broadcasting.
    Usage:
      # for y = func(x)
      y_list = ArrayFunc(func)(x_list)
    Could be used as a decorator.
    """
    def broadcasted_func(x_list):
        if isinstance(x_list, np.ndarray):
            y_list = np.zeros(x_list.shape)
        elif isinstance(x_list, (list, ArrayfyList)):
            y_list = [None] * len(x_list)
        else:
            # not a list, assume scalar
            return func(x_list)
        # TODO: maybe use parallel here
        for j, x in enumerate(x_list):
            y_list[j] = func(x)
        return y_list
    return broadcasted_func

def inject_swc_utils(ns, oracle = None):
    """
The following variables are prepared:
    gui_ctrl, iren, interactor, ren, swcs
See the help like `help(swcs)`, or reference the plugins directory.
    """
    if oracle is None:
        # e.g. when used in UIActions and passing ns = locals()
        oracle = ns['self']
    ns |= globals() | ns   # merge globals in utils.py but not overwrite ns.
    ns['gui_ctrl']   = oracle.gui_ctrl
    ns['iren']       = oracle.iren
    ns['interactor'] = oracle.interactor
    ns['ren']        = oracle.GetRenderers(1)

    gui_ctrl = ns['gui_ctrl']
    iren = ns['iren']
    ns.update(NamespaceOfSwcUtils(gui_ctrl, iren))

def NamespaceOfSwcUtils(gui_ctrl, iren):
    ns = {}
    swc_objs = list(gui_ctrl.GetObjectsByType('swc').values())
    ns['swcs'] = ArrayfyList(swc_objs)
    ns['Render'] = iren.GetRenderWindow().Render
    return ns