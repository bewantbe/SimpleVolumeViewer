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

    tr = trs.forest_to_tree()[0]
    tr.normalize()

    # find (guess) all axon of a neuron tree
    def mark_axon(tr):
        # TODO: impliment it.
        # find most far leaf. alt. most far point
        leaves_id  = tr.leaves_id()
        leaves_pos = tr.id_coordinate(leaves_id)
        root_pos   = tr.id_coordinate(0)
        idx_max_dis = np.argmax(VecNorm(leaves_pos - root_pos))
        farend_id  = leaves_id[idx_max_dis]
        # find root branch of axon
        rpath      = tr.root_path(farend_id)
        subtree_id = tr.subtree_id(rpath[1])
        # mark axon
        tr.node_type[subtree_id] = 2
        # mark dentrite
        ids_den = np.ones(len(tr), dtype=bool)
        ids_den[subtree_id] = 0
        ids_den[0] = 0
        tr.node_type[ids_den] = 1
        # axon sub-tree
        tr_axon = tr[np.stack([0], subtree_id)]
        return tr_axon

    def make_densor_tree(tr, dist_max):
        # implimented.
        return tr_new

    def aggregated_nodes(tr_axon, r_tol):
        """Aggregated node: a node that has non-the-same branch neighbors"""\
        """within r_tol. The more neighbors, the heavier the aggregation."""\
        """Need to densify first to get a fair result."""
        # TODO: impliment it.
        # Algo. kd-tree
        return ids_aggregated, vec_n_neighbor

    def mapping_to_densor_area(tr_axon, ids_rich):
        # TODO: impliment it.

    def overlapping_of_Aggregation_area



    print('Done.')
