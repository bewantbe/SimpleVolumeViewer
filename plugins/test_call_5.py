#

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import neu3dviewer.utils
from neu3dviewer.utils import (
    ArrayfyList,
    ArrayFunc,
    NamespaceOfSwcUtils,
)

if 'fiber-inspector' not in sys.path:
    sys.path.insert(1, '../fiber-inspector')
from find_consensus import *
from batch_swc_stat import List, _ai, _af

def PluginMain(ren, iren, gui_ctrl):
    neu3dviewer.utils.debug_level = 3
    user_ns = NamespaceOfSwcUtils(gui_ctrl, iren)
    swcs = user_ns['swcs']
    print('Number of SWC:', len(swcs))

    # find axon (by length to root?)

    print('Done.')
