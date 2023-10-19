#!/usr/bin/env python3

# in ipython
# %run /mnt/data_ext/swc_collect/RM009_manrefine/extract_mody_pos.py

import sys
import re
import numpy as np
import pandas as pd
sys.path.append('/home/xyy/code/fiber-inspector/')
from find_consensus import load_fiber_transformed_coor_net

root_dir = '/mnt/data_ext/swc_collect/RM009_manrefine/'
xlsx_path = '神经元审查表格.xlsx'

#data_frame = pd.read_excel(root_dir + xlsx_path)
#data_frame = pd.read_excel(file_path, sheet_name=sheet_name)
# return pandas.core.frame.DataFrame

xls = pd.ExcelFile(root_dir + xlsx_path)
sheet_names = xls.sheet_names

def find_name_in_list_index(substr, name_list):
    if len(name_list) == 0:
        return ()
    idx = np.flatnonzero(np.char.find(name_list, substr) != -1)
    return idx

def find_name_in_list(idx, name_list):
    idx = find_name_in_list_index(str(idx)+'号', name_list)
    if len(idx)>0:
        return name_list[idx[0]]
    else:
        return ()

neuron_sheet_list = sheet_names[1:-1]
# get numerical ids in neuron_sheet_list
neuron_ids = []
for neuron_sheet_name in neuron_sheet_list:
    m = re.match(r'^\d+', neuron_sheet_name)
    neuron_ids.append(int(m.group(0)))

version_list = ['v1.6.6', 'v1.7.0', 'v1.7.1', 'v1.7.2']

pos_by_neuron = [None] * len(neuron_ids)

# loop over neuron ids
for idx_neu, neuron_id in enumerate(neuron_ids):
    print('Processing neuron id:', neuron_id)
    neuron_sheet_name = find_name_in_list(neuron_id, sheet_names)
    neuron_sheet = xls.parse(neuron_sheet_name)
    if neuron_sheet['节点'].dtype == np.float64:
        node_list = [f'{s:.0f}' for s in neuron_sheet['节点'].tolist()]  # force to string
    else:
        node_list = [str(s) for s in neuron_sheet['节点'].tolist()]

    # break the node_list by the keyword 查
    idx_sep = list(find_name_in_list_index('查', node_list))
    idx_sep.insert(0, 0)
    idx_sep.append(len(node_list))
    node_bucket_by_version = []
    for i in range(len(idx_sep)-1):
        s = node_list[idx_sep[i]:idx_sep[i+1]]
        # remove strings in s that are not a number
        s = [int(x) for x in s if x.isdigit()]
        node_bucket_by_version.append(s)

    # loop over node_bucket_by_version
    node_pos_by_version = [None] * len(node_bucket_by_version)
    for j, node_bucket in enumerate(node_bucket_by_version):
        print('  version', version_list[j])
        # load the neuron lyp file
        lyp_path = root_dir + version_list[j] + '/neuron#' + str(neuron_id) + '.lyp'
        fb = load_fiber_transformed_coor_net(lyp_path)
        # construct dict from node id to idx in the array
        reverse_map = dict(zip(
            [i[0] for i in fb[0]],
            range(len(fb[0]))
        ))
        print('  Number of nodes:', len(node_bucket))
        # check non-exist nodes
        non_exist_nodes = [i for i in node_bucket if i not in reverse_map.keys()]
        if len(non_exist_nodes) > 0:
            print('  Non-exist nodes:', non_exist_nodes)
        # get all node position from the node id
        node_pos_by_version[j] = [fb[1][reverse_map[i]] for i in node_bucket
                                  if i in reverse_map.keys()]

    pos_by_neuron[idx_neu] = node_pos_by_version

# merge all pos
pos_all = []
for neu in range(len(pos_by_neuron)):
    for ver in range(len(pos_by_neuron[neu])):
        pos_all.extend(pos_by_neuron[neu][ver])

va = np.vstack(pos_all)

if 0:
    import matplotlib.pyplot as plt
    plt.ion()

    # show 3D position
    fig = plt.figure(9)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(va[:,0], va[:,1], va[:,2], marker='o')

    # show hist in z direction
    plt.figure(12)
    #plt.cla()
    a = va[:,2] % 300.0
    a = np.where(a > 150, a - 300, a)
    plt.hist(a, 75)
    plt.title('wrong nodes')
    plt.xlabel('distance to inter-slice surface')
    plt.ylabel('hist (count)')

if 1:
    # set colors of the fibers
    swcs.color = 'grey'
    swcs.opacity = 0.5

    # add points of errors
    obj_conf = \
    {
        "type": "Sphere",
    }
    for i, v in enumerate(va):
        print(i, v)
        objname = 'pt'+str(i)
        gui_ctrl.AddObject(objname, obj_conf)
        o = gui_ctrl.scene_objects[objname]
        o.position = v
        o.color = 'green'
        o.actor.SetScale(100.0, 100.0, 100.0)

if 0:
    # rescale spheres in the scene
    for i in enumerate(va):
        print(i, v)
        objname = 'pt'+str(i)
        o = gui_ctrl.scene_objects[objname]
        o.actor.SetScale(100.0, 100.0, 100.0)

if 0:
    ss = 10.0
    # rescale spheres in the scene
    for k, conf in gui_ctrl.scene_saved['objects'].items():
        if not k.startswith('pt'):
            continue
        print(k)
        o = gui_ctrl.scene_objects[k]
        o.actor.SetScale(ss, ss, ss)

if 0:
    import matplotlib.pyplot as plt
    plt.ion()

    # show hist in z direction
    plt.figure(12)
    #plt.cla()
    a = gui_ctrl.point_set_holder().T[:,2] % 300.0
    a = np.where(a > 150, a - 300, a)
    plt.hist(a, 75)
    plt.title('all nodes')
    plt.xlabel('distance to inter-slice surface')


