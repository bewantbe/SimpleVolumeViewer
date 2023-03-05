# Run by:
# python3 -m pytest tests

from neu3dviewer.ui_interactions import *
import numpy as np
from numpy.random import rand

f32 = np.float32

def test_point_set_holder():
    # usual condition
    psh = PointSetHolder()
    ps = [rand(3, 5).astype(f32), rand(3, 11).astype(f32), rand(3, 7).astype(f32)]
    psh.AddPoints(ps[0], 'a')
    psh.AddPoints(ps[1], 'b')
    psh.AddPoints(ps[2], 'c')
    psh.RemovePointsByName('b')
    assert np.all(psh._points_list[0][:, 5:] == ps[2])
    assert len(psh) == 5+7
    assert psh._point_set_boundaries == [0, 5, 5+7]
    assert psh.name_list == ['a', 'c']
    assert psh.name_idx_map['c'] == 1
    
    # non-exist
    psh.RemovePointsByName('d')
    assert len(psh._points_list) == 1
    assert len(psh) == 5+7
    assert len(psh.name_list) == 2

    # full-remove
    psh.RemovePointsByName(['c', 'a'])
    assert psh._points_list[0].shape[1] == 0
    assert psh._point_set_boundaries == [0]
    assert len(psh) == 0
    assert len(psh.name_list) == 0

    # empty
    psh = PointSetHolder()
    psh.RemovePointsByName('b')
    assert len(psh._points_list) == 0
    assert len(psh) == 0
    assert psh._point_set_boundaries == [0]
    assert psh.name_list == []

