#
import numpy as np

from vtkmodules.vtkCommonDataModel import (
    vtkColor3ub,
    vtkColor3d
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
)

debug_level = 5

def dbg_print(level, *p, **keys):
    """
    Used for printing error and debugging information.
    Controlled by global (module) debug_level.
    Higher debug_level will show more infomation.
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

def str2array(s):
    """ Convert list of numbers in string form to `list`. """
    if not isinstance(s, str):
        return s
    if s[0] == '[':
        # like '[1,2,3]'
        v = [float(it) for it in s[1:-1].split(',')]
    else:
        # like '1 2 3'
        v = [float(it) for it in s.split(' ')]
    return v

def _mat3d(d):
    """ Convert vector of length 9 to 3x3 numpy array. """
    return np.array(d, dtype=np.float64).reshape(3,3)

def vtkGetColorAny(c):
    if isinstance(c, str):
        colors = vtkNamedColors()
        return colors.GetColor3d(c)
    elif isinstance(c, vtkColor3ub):
        return vtkColor3d(c[0]/255, c[1]/255, c[2]/255)
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
    """ Return a name start with prefix but not occured in name_set. """
    i = 1
    name = prefix
    while name in name_set:
        name = prefix + '.%.3d'%i
        i += 1
    return name

def MergeFullDict(d_contain, d_update):
    """
    Update dict d_contain by d_update.
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
                else:  # overwirte
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

