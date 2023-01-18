#

import numpy as np
from numpy import array as _a
from numpy.linalg import norm as norm
import matplotlib as mpl
import matplotlib.pyplot as plt

def normalize(r):
    l = norm(r)
    if l == 0:
        return r
    return r / l

def value_normalize(r):
    max_val = np.max(r)
    min_val = np.min(r)
    if max_val == min_val:
        return r
    return (r - min_val) / (max_val - min_val)

def GetColorMapFunction(cmap_name):
    """
    Use it like:
        cmap_f = GetColorMapFunction('plasma')
        print('Colors:', cmap_f(0.0), cmap_f(0.5), cmap_f(1.0))
    """
    cmap_table = plt.get_cmap(cmap_name)
    if isinstance(cmap_table, mpl.colors.LinearSegmentedColormap):
        return cmap_table
    cmap_f = mpl.colors.LinearSegmentedColormap.from_list(
                 cmap_name + '_s',
                 plt.get_cmap(cmap_name).colors)
    return cmap_f

def GetValueByAnchorPoints(color_anchor_points, swc_objs):
    # color_anchor_points contains three points:
    # [left anchor, right anchor, backward anchor(for direction)]
    # Value(left anchor)  = 0.0,
    # Value(right anchor) = 1.0,
    # the three points in `color_anchor_points` denote a plane A,
    # which the normal of A is parallel to the cutting plane B
    # which the plane B also pass through left and right anchor points.
    
    color_direction = color_anchor_points[1] - color_anchor_points[0]
    color_direction_length = norm(color_direction)
    color_direction = normalize(color_direction)
    # cutting color plane
    plane_normal = \
        np.cross(
            normalize(np.cross(
                color_direction, 
                color_anchor_points[2] - color_anchor_points[0]
            )),
            color_direction)
    plane_origin = color_anchor_points[0]
    
    n_swc = len(swc_objs)
    s_v = np.zeros(n_swc)
    for k, o in enumerate(swc_objs):
        v = int(o.swc_name.split('#')[1]) / n_swc
        
        r_xyz = o.tree_data[1][:,0:3]
        
        # find a point that is near the cutting plane
        near_plane_idx = np.argmin(np.abs(
                             (r_xyz - plane_origin).dot(plane_normal)
                         ))
        v = (r_xyz[near_plane_idx] - plane_origin).dot(color_direction) / color_direction_length
        print('f =', o.swc_name, '  v =', v)  # , '  color =', c
        s_v[k] = v
    return s_v

def PluginMain(ren, iren, gui_ctrl):
    swc_objs = gui_ctrl.GetObjectsByType('swc')
    print('Number of SWC:', len(swc_objs))

    n_swc = len(swc_objs)
    cmap_f = GetColorMapFunction('plasma')
    #cmap_f = GetColorMapFunction('gray')
    
    color_anchor_points = _a([[5008.4, 2742.5, 4912.], [5229.2, 3079.6, 5099.6], [5070., 3148.8, 4702.8]])
    s_v = GetValueByAnchorPoints(color_anchor_points, swc_objs)
    s_v = value_normalize(s_v)
    
    for k, o in enumerate(swc_objs):
        c = cmap_f(s_v[k])
        o.color = c

#    plt.ioff()
#    plt.hist(s_v, 100)
#    plt.show()
#    print('hello?')

    iren.GetRenderWindow().Render()
    # TODO: make it possible to open a ipython environment, like Matlab.
