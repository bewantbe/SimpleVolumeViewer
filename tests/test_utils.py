#!/usr/bin/env python3
# test_data_loader.py

# run by
# python3 -m pytest tests/test_utils.py

import pytest

from neu3dviewer.utils import *

def test_ArrayFunc():
    fn = lambda a: a**2
    x = [1,2,3,4]
    y0 = list(map(fn, x))

    y1 = ArrayFunc(fn)(x)
    assert y0 == y1

    y2 = ArrayFunc(fn, n_parallel = 2, progress=True)(x)
    assert y1 == y2
