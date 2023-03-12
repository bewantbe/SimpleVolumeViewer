#!/usr/bin/env python3
# test_data_loader.py

# run by
# python3 -m pytest tests/test_data_loader.py

import pytest

from neu3dviewer.data_loader import *

dir_path = 'tests/ref_data/swc_ext/'

def test_swc_import():
    filepath = dir_path + 't3.3.swc'

    tr = LoadSWCTree(filepath)
    assert tr[0].shape == (16,3)
    assert tr[1].shape == (16,4)

    ps = SplitSWCTree(tr)
    assert len(ps) == 10

    dp = SimplifyTreeWithDepth(ps, output_mode = 'depth')
    assert max(dp) == 3
    assert len(dp) == len(ps)
