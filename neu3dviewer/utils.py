# SPDX-License-Identifier: GPL-3.0-or-later

# Small utility functions

import sys
import datetime
import numpy as np
from numpy import eye, sin, cos
import pprint

# for array function
from multiprocessing import Pool
from multiprocessing import cpu_count

import joblib

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

def UpdateTransferFunction(obj_prop, prop_conf):
    dbg_print(4, 'UpdateTransferFunction():')
    # both obj_prop and prop_conf will be updated
    if 'opacity_transfer_function' in prop_conf:
        otf_conf = prop_conf['opacity_transfer_function']
        if 'opacity_scale' in otf_conf:
            otf_s = otf_conf['opacity_scale']
            UpdatePropertyOTFScale(obj_prop, otf_s)
        else:
            pf = obj_prop.GetScalarOpacity()
            pf.RemoveAllPoints()
            for v in otf_conf['AddPoint']:
                pf.AddPoint(*v)
            if hasattr(obj_prop, 'prop_conf'):
                obj_prop.prop_conf['opacity_transfer_function']['AddPoint'] \
                    = otf_conf['AddPoint']
    if 'color_transfer_function' in prop_conf:
        ctf_conf = prop_conf['color_transfer_function']
        if 'trans_scale' in ctf_conf:
            ctf_s = ctf_conf['trans_scale']
            UpdatePropertyCTFScale(obj_prop, ctf_s)
        else:
            ctf = obj_prop.GetRGBTransferFunction()
            ctf.RemoveAllPoints()
            for v in ctf_conf['AddRGBPoint']:
                ctf.AddRGBPoint(*v)
            if hasattr(obj_prop, 'prop_conf'):
                obj_prop.prop_conf['color_transfer_function']['AddRGBPoint'] \
                    = ctf_conf['AddRGBPoint']

# copy from readtiff.py in 3dimg_cruncher
# thanks https://stackoverflow.com/a/61343915/4620849
def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

# copy from readtiff.py in 3dimg_cruncher
def img_basic_stat(imgarr):
    imgarr = imgarr.flatten()

    x_max = np.max(imgarr)
    bin_cnt = np.bincount(imgarr)
    v_perce = list(weighted_percentile(np.arange(x_max+1), bin_cnt,
                                       [0.001, 0.01, 0.5, 0.99, 0.999]))
    stat = {
        'min'    : np.min(imgarr),
        'max'    : x_max,
        'mean'   : np.mean(imgarr),
        'median' : np.median(imgarr),
        'n_unique_value' : np.sum(bin_cnt>0), # PS: np.unique is much slower
        'q0.001' : v_perce[0],
        'q0.01'  : v_perce[1],
        'q0.5'   : v_perce[2],
        'q0.99'  : v_perce[3],
        'q0.999' : v_perce[4],
        #'q0.001, q0.5, q0.999' : list(np.quantile(imgarr, [0.001, 0.5, 0.999]))
    }
    #pprint.pprint(stat)
    return stat

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

def RotationMat(theta, axis :int):
    # construct a rotation matrix around the axis
    # or try
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    #from scipy.spatial.transform import Rotation
    s = sin(theta)
    c = cos(theta)
    m = eye(3)
    if axis == 2 or axis == 'z':  # z -> [x, y]  (2 -> [0, 1])
        ax = 0
        ay = 1
        az = 2
    elif axis == 1 or axis == 'y':  # y -> [z, x]  (1 -> [2, 0])
        ax = 2
        ay = 0
        az = 1
    elif axis == 0 or axis == 'x':  # x -> [y, z]  (0 -> [1, 2])
        ax = 1
        ay = 2
        az = 0
    else:
        raise ValueError('invalid axis')
    m[ax,ax] = m[ay,ay] = c
    m[ax,ay] = -s
    m[ay,ax] = s
    return m

def VecNorm(x, axis=0):
    return np.sqrt(np.sum(x*x, axis=axis))

class Struct:
    """
    Somewhat like the struct in matlab.
    
    Usage example:
      s = Struct(a=1, b=2)
      s.c = 3
      print(s)
    """
    def __init__(self, **kv):
        for k, v in kv.items():
            setattr(self, k, v)

    def __repr__(self):
        s = '<Struct>(\n' \
          + '\n'.join([
              f'  .{k} = ' + v.__repr__()
              for k, v in vars(self).items()
            ]) \
          + '\n)'
        return s

def get_num_in_str(a):
    # Return what integer part in the string (usually filename)
    #   a = 'fae#323-23.swc' is invalid, exception
    #   a = 'fae#323.swc' is valid, return 323
    u = np.array(bytearray(a, encoding='utf-8'))
    # consist of '0123456789'
    is_num = (48 <= u) & (u <= 57)
    start_idx = is_num.argmax()
    end_idx = len(is_num) - is_num[start_idx:][::-1].argmax()
    #print('=', a[start_idx:end_idx])
    return int(a[start_idx:end_idx])

def contain_int(a):
    ok = True
    try:
        u = get_num_in_str(a)
    except ValueError:
        ok = False
    return ok

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

def ShowTidyArrayRepr(list_type, obj_list, cr = '\n'):
    show_li = lambda idxs, sep: sep.join([f'{obj_list[j]}' for j in idxs])
    n = len(obj_list)
    n_head = 5
    n_tail = 3
    s = list_type
    if n > n_head + n_tail:
        head = show_li(range(n_head), ','+cr)
        tail = show_li(range(n-n_tail,n), ','+cr)
        s += '['+cr + head + ','+cr+' ... ,'+cr + tail + cr+']'
    else:
        s += '[' + show_li(range(n), ', ') + ']'
    return s

def bind_property_broadcasting(li, pn, docs):
    """
    Make the pn broadcast to all items in li.
    li: the array-like class to bind to.
    pn: the property (function) to be bind.
    """
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
    def __init__(self, obj_list, index_list = None):
        self.index_style = 'numeric'
        if isinstance(obj_list, dict):
            e_obj_list = obj_list
            obj_list = list(e_obj_list.values())
            index_list = list(e_obj_list.keys())
            self.index_style = 'dict'
        if not isinstance(obj_list, (list, np.ndarray, tuple)):
            raise TypeError('Wrapper for list only.')
        self.obj_list = obj_list
        self.index_list = index_list
        setattr(self, '__iter__', obj_list.__iter__)
        #setattr(self, '__len__',  obj_list.__len__)  # need to rebind
        self.rebuild_index()
        self._bind_properties()

    def list(self):
        return self.obj_list

    def rebuild_index(self):
        """
        Rebuild indexing string according style.
        index_style can be 'numeric'
        """
        index_style = self.index_style
        # guess the type
        if self.__len__() == 0:
            self.obj_dict = {}
            return
        
        f_extract_num = lambda a: a.lstrip()
        
        # contruct fn(), which give the index itself.
        s = self.obj_list[0]
        if hasattr(s, 'swc_name') and  index_style != 'dict':
            # should be a swc file
            if index_style == 'numeric':
                # try extract the numerical part.
                if np.all(np.array([contain_int(o.swc_name) for o in self.obj_list], dtype=bool)):
                    fn = lambda j, o: str(get_num_in_str(o.swc_name))
                else:  # give up
                    dbg_print(2, 'ArrayfyList::rebuild_index(): Not numeric indexable. Use swc name instead')
                    fn = lambda j, o: o.swc_name
            elif index_style == 'file_name':
                fn = lambda j, o: o.swc_name
            else:
                raise TypeError(f'Unknown index style "{index_style}".')
        else:
            if self.index_list is not None:
                if len(self.index_list) != len(self.obj_list):
                    raise IndexError("Length does not match.")
                fn = lambda j, o: self.index_list[j]
            else:
                fn = lambda j, o: str(j)

        self.obj_dict = {fn(j, o): o for j, o in enumerate(self.obj_list)}
    
    def _bind_properties(self):
        if len(self.obj_list) == 0:
            return
        # the items has properties
        s = self.obj_list[0]
        if not hasattr(s, '__dict__'):
            return
        ty_s = type(s)
        attribute_names = vars(s).keys()  # might use dir(s) to includes more
        # list all attributes typed property() and in the form "_prop"
        prop_names = [k[1:] for k in attribute_names \
                       if k.startswith('_') and \
                         not k.startswith('__') and \
                         isinstance(getattr(ty_s, k[1:], None), property)]
        for pn in prop_names:
            bind_property_broadcasting(self, pn, getattr(ty_s, pn).__doc__)

        # list all attributes of pure data item
        ditem_names = [k for k in attribute_names \
                       if not k.startswith('_') and \
                         not hasattr(getattr(s, k, None), '__call__')]
        #dbg_print(5, '_bind_properties(): converting these:', ditem_names)
        for pn in ditem_names:
            bind_property_broadcasting(self, pn, 'broadcasted' + pn)

    def __len__(self):
        return self.obj_list.__len__()

    def keys(self):
        return self.obj_dict.keys()

    def items(self):
        return self.obj_dict.items()

    def __repr__(self):
        return ShowTidyArrayRepr(
            self.__class__.__name__,
            self.obj_list)

    def __str__(self):
        return ShowTidyArrayRepr(
            self.__class__.__name__,
            list(self.obj_dict.keys()),
            chr(10))

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # index by swc name, like "['123']"
            return self.obj_dict[idx]
        elif isinstance(idx, slice):
            # indexing by slice or range, like: "[:10]"
            return ArrayfyList(self.obj_list[idx])
        elif isinstance(idx, (list, np.ndarray, type(self))):
            if hasattr(idx, 'dtype') and (idx.dtype == bool):
                # '[[True, False, ...]]'
                assert len(self) == len(idx)
                return ArrayfyList([
                    self.__getitem__(i) 
                    for i in range(len(self))
                    if idx[i]
                ])
            else:
                # array style index, like '[["1", "2", "3"]]'
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
            if hasattr(idx, 'dtype') and (idx.dtype == bool):
                # '[[True, False, ...]]'
                idx_t = np.flatnonzero(idx)
                if not hasattr(val, '__len__'):
                    val = [val] * len(idx_t)
                assert len(idx_t) == len(val)
                for j, i in enumerate(idx_t):
                    self.obj_list[i] = val[j]
            else:
                # array style index, like "[[1, 2, 3]]"
                for j, i in enumerate(idx):
                    self.obj_list[i] = val[j]
        else:
            # index by order in the list, like "[123]"
            self.obj_list[idx] = val
        self.rebuild_index()

def ArrayFunc(func, n_parallel = 1, progress = False):
    """
    Arrayfy the func such that it accept array(list) input, i.e. broadcasting.

    Tips for n_parallel:
        For very simple function, e.g. x**2, use n_parallel>1 will make
        things 10x times slower, and larger n_parallel will make it worse.
        Set n_parallel = -1 to use all CPUs.

    Usage:
      # for y = func(x)
      y_list = ArrayFunc(func)(x_list)
    Could be used as a decorator.

    See also: np.vectorize, map
    """
    if progress:
        def func_idx(x, j):
            print('job =', j)
            return func(x)
    else:
        func_idx = lambda x, j: func(x)
    
    if n_parallel == 1:
        def broadcasted_func(x_list):
            if isinstance(x_list, np.ndarray):
                y_list = np.zeros(x_list.shape)
            elif isinstance(x_list, (list, ArrayfyList)):
                y_list = [None] * len(x_list)
            else:
                # not a list, assume scalar
                return func(x_list)
            out = False  # it is a function, not command
            for j, x in enumerate(x_list):
                y_list[j] = func_idx(x, j)
                out |= y_list[j] is not None
            if out:
                return y_list
    elif isinstance(n_parallel, str) and \
         n_parallel.startswith('multiprocessing'):
        # e.g. n_parallel = 'multiprocessing:4'
        n_parallel = int(n_parallel.split(':')[1])
        def broadcasted_func(x_list):
            with Pool(n_parallel) as p:
                # TODO: Bug: Error: PicklingError: Can't pickle
                y_list = p.starmap(func_idx, zip(x_list, range(len(x_list))))
            out = False
            for y in y_list:
                out |= y is not None
            if out:
                return y_list
    else:
        def broadcasted_func(x_list):
            y_list = joblib.Parallel(n_jobs = n_parallel) \
                (joblib.delayed(func_idx)(x, j)
                    for x, j in zip(x_list, range(len(x_list))))
            out = False
            for y in y_list:
                if y is not None:
                    out = True
                    break
            if out:
                return y_list

    return broadcasted_func

def inject_swc_utils(ns, oracle = None):
    """The following variables are prepared:\n""" \
    """    swcs, gui_ctrl, iren, interactor, ren\n""" \
    """See the help like `help(swcs)`, or reference the plugins directory."""
    if oracle is None:
        # e.g. when used in UIActions and passing ns = locals()
        oracle = ns['self']
    ns |= globals() | ns   # merge globals in utils.py but not overwrite ns.
    ns['gui_ctrl']   = oracle.gui_ctrl
    ns['iren']       = oracle.iren
    ns['interactor'] = oracle.interactor
    ns['ren']        = oracle.GetRenderers(1)
    ns['norm']       = np.linalg.norm

    gui_ctrl = ns['gui_ctrl']
    iren = ns['iren']
    ns.update(NamespaceOfSwcUtils(gui_ctrl, iren))

def NamespaceOfSwcUtils(gui_ctrl, iren):
    ns = {}
    swc_obj_dict = gui_ctrl.GetObjectsByType('swc')
    swc_objs = list(swc_obj_dict.values())
    ns['swcs'] = ArrayfyList(swc_objs)
    ns['map_swc_id'] = ArrayfyList(list(range(len(swc_obj_dict))),
                                   tuple(swc_obj_dict.keys()))
    ns['Render'] = iren.GetRenderWindow().Render
    
    # Tips(tricks):
    # Batch get swc name from internal object id:
    #     swcs[map_swc_id[['swc.766', 'swc.452', 'swc.953']]].swc_name
    # swcs[map_swc_id[gui_ctrl.selected_objects]].swc_name

    return ns

def IPython_embed(*, header="", compile_flags=None, **kwargs):
    """
    Adapted from IPython/terminal/embed.py version 8.11.0.
    License: BSD License (BSD-3-Clause)
    
    See `help(IPython.embed)` for original help.

    Modification: Add line magic "autoreload" for convenience.
    
    It is adapted here to bypass the InteractiveShellEmbed bug in 
    IPython v8.11 and also to add autoreload magic.
    """
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.terminal.ipapp import load_default_config
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    
    config = kwargs.get('config')
    if config is None:
        config = load_default_config()
        # this config won't work as usual, thus we use run_line_magic
        #config.InteractiveShellApp.exec_lines = [
        #    '%reload_ext autoreload',
        #    '%autoreload 2'
        #]
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config
    using = kwargs.get('using', 'sync')
    if using :
        kwargs['config'].update({
            'TerminalInteractiveShell':{
                'loop_runner' : using,
                'colors' : 'NoColor',
                'autoawait': using!='sync'
            }
        })
    #save ps1/ps2 if defined
    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass
    #save previous instance
    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
    frame = sys._getframe(1)
    shell = InteractiveShellEmbed.instance(_init_location_id='%s:%s' % (
        frame.f_code.co_filename, frame.f_lineno), **kwargs)

    shell.run_line_magic('reload_ext', 'autoreload')
    shell.run_line_magic('autoreload', '2')

    shell(header=header, stack_depth=2, compile_flags=compile_flags,
        _call_location_id='%s:%s' % (frame.f_code.co_filename, frame.f_lineno))
    InteractiveShellEmbed.clear_instance()
    #restore previous instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
        for subclass in cls._walk_mro():
            subclass._instance = saved_shell_instance
    if ps1 is not None:
        sys.ps1 = ps1
        sys.ps2 = ps2

