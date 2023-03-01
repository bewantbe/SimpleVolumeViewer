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

    ntrees = [ntree((s.tree_swc[0], s.tree_swc[1][:,:3])) for s in swcs]
    ntrees = ArrayfyList(ntrees)
    
    # get stats
    nts_stat = List(ArrayFunc(lambda s: s.stat(False))(ntrees))

    if 1:
        distr_branch = _ai(nts_stat['n_branches'])
        # seems  0.0 ~ 0.3 of branches do not contain much axon
        v1_branch = np.quantile(distr_branch, 0.3)
        v2_branch = np.quantile(distr_branch, 0.99)

        swcs.visible = False
        swcs[(distr_branch > v1_branch) & (distr_branch < v2_branch)].visible = True
    else:
        distr = _ai(nts_stat['geo_len'])
        # seems  0.0 ~ 0.3 of branches do not contain much axon
        v1 = np.quantile(distr, 0.2)
        v2 = np.quantile(distr, 0.3)

        swcs.visible = False
        swcs[(distr > v1) & (distr < v2)].visible = True
    
    print('Done.')
