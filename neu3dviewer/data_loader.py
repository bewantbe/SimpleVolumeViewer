# SPDX-License-Identifier: GPL-3.0-or-later

# data loader that do not involve too much of VTK

import os.path
import pprint
import json
import time
# hope it is better than list. https://docs.python.org/3/library/array.html
import array

import numpy as np
from numpy import sqrt, sin, cos, tan, pi
from numpy import array as _a
dtype_id = np.int32
dtype_coor = np.float32
np_typecodes_map = {np.float32:'f', np.float64:'d', np.int32:'l', np.int64:'q'}

import scipy.sparse

import tifffile
# ondemond load, reduce load time, reduce message in logger
#import h5py
#import zarr

from .utils import (
    dbg_print,
    str2array,
    array2str,
    slice_from_str
)

from vtkmodules.vtkIOImage import (
    vtkImageImport
)

def read_tiff(tif_path, as_np_array = True):
    """
    Read tiff file, return images (as np.ndarray) and meta data.
    Copy from volumeio.py
    See also https://pypi.org/project/tifffile/
    """
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_val.name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    if as_np_array:
        images = tifffile.imread(tif_path)
    else:
        images = []
        for page in tif.pages:
            images.append(page.asarray())

    # TODO: determine oblique_image more properly
    if ('oblique_image' not in metadata) and len(images) > 0:
        corner_vals = _a([[[images[ii][jj,kk]
                            for ii in [0,-1]]
                            for jj in [0,-1]]
                            for kk in [0,-1]]).flatten()
        is_tilted = np.all(corner_vals > 0)
        metadata['oblique_image'] = (metadata['ImageLength']==788) and is_tilted
        if (not is_tilted) and ('imagej' in metadata) \
                           and (metadata['imagej'] is not None)\
                           and ('voxel_size_um' not in metadata['imagej']):
            metadata['imagej']['voxel_size_um'] = (1.0, 1.0, 2.5)

    return images, metadata

def read_tiff_meta(tif_path):
    """
    Read tiff file, return image metadata.
    See also https://pypi.org/project/tifffile/
    """
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    
    metadata['n_pages'] = len(tif.pages)
    return metadata

def read_ims(ims_path, extra_conf = {}, cache_reader_obj = False):
    """
    Read Imaris compatible (HDF5) image file.
    Return image array and metadata.
    """
    import h5py         # will only load at the first time

    dbg_print(4, 'read_ims(): extra_conf =', extra_conf)
    dim_ranges = slice_from_str(str(extra_conf.get('range', '[:,:,:]')))
    dbg_print(4, '  Requested dim_range:', dim_ranges)
    
    # TODO: how to implement cache_reader_obj?
    ims = h5py.File(ims_path, 'r')
    level      = int(extra_conf.get('level', 0))
    channel    = int(extra_conf.get('channel', 0))
    time_point = int(extra_conf.get('time_point', 0))
    img = ims['DataSet']['ResolutionLevel %d'%(level)] \
                        ['TimePoint %d'%(time_point)] \
                        ['Channel %d'%(channel)]['Data']

    dbg_print(4, '  Done image selection. Shape: ', img.shape, ' dtype =', img.dtype)

    # convert metadata in IMS to python dict
    metadata = {'read_ims':
        {'level': level, 'channel': channel, 'time_point': time_point}}
    if 'DataSetInfo' in ims:
        img_info = ims['DataSetInfo']
        for it in img_info.keys():
            metadata[it] = \
                {k:''.join([c.decode('utf-8') for c in v])
                    for k, v in img_info[it].attrs.items()}

    t0 = time.time()
    img_clip = np.array(img[dim_ranges])         # actually read the data
    dbg_print(4, 'read_ims(): img read time: %6.3f sec.' % (time.time()-t0))
    #img_clip = np.transpose(np.array(img_clip), (2,1,0))

    # TODO: find correct voxel size and whether it is oblique.
    metadata['imagej'] = {'voxel_size_um': '(1.0, 1.0, 1.0)'}
    b_fmost = False
    if b_fmost:
        #l0 = _a([0.35, 0.35, 1.0])
        l0 = _a([1.0, 1.0, 1.0])
        lsize = tuple(l0 * (2**level))
        metadata['imagej'] = {'voxel_size_um': lsize}
    metadata['oblique_image'] = False

    return img_clip, metadata

def read_zarr(zarr_path, extra_conf = {}, cache_reader_obj = False):
    """Read zarr from img_path, return image array and metadata."""
    import zarr     # will only actually load at the first time

    dbg_print(4, 'read_zarr(): extra_conf =', extra_conf)
    str_ranges = str(extra_conf.get('range', '[:,:,:]'))
    dim_ranges = slice_from_str(str_ranges)
    dbg_print(4, '  Requested dim_range:', dim_ranges)

    z = zarr.open(zarr_path, mode='r')
    t0 = time.time()
    img_clip = np.array(z[dim_ranges])         # actually read the data
    # we use xyz order in zarr, but in the rest of the program we use zyx
    img_clip = img_clip.T
    dbg_print(4, 'read_zarr(): img read time: %6.3f sec.' % (time.time()-t0))

    metadata = {'zarr': {'range': str_ranges}}
    metadata['zarr'].update(z.attrs.asdict())
    metadata['imagej'] = {'voxel_size_um': '(1.0, 1.0, 1.0)'}
    metadata['oblique_image'] = False

    return img_clip, metadata

def Read3DImageDataFromFile(file_name, *item, **keys):
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        img_arr, img_meta = read_tiff(file_name)
    elif file_name.endswith('.ims') or file_name.endswith('.h5'):
        img_arr, img_meta = read_ims(file_name, *item, **keys)
    elif os.path.isdir(file_name) and os.path.isfile(file_name + '/.zarray'):
        img_arr, img_meta = read_zarr(file_name, *item, **keys)
    else:
        raise TypeError('File format not supported: ' + file_name)
    dbg_print(5, pprint.pformat(img_meta))
    return img_arr, img_meta

def Save3DImageToFile(file_name, img_arr, img_meta):
    img_arr = img_arr[:, np.newaxis, :, :]
    voxel_size_um = (1.0, 1.0, 1.0)
    tifffile.imwrite(file_name, img_arr,
                     imagej=True,
                     #compression='zlib', compressionargs={'level': 8},
                     compression=['zlib', 2],
                     resolution=(1/voxel_size_um[0], 1/voxel_size_um[1]), 
                     metadata={'spacing': voxel_size_um[2], 'unit': 'um', 
                               **img_meta})

def ImportImageArray(img_arr, img_meta):
    """
    Import image array to vtkImageImport() to have a connection.
    Input:
        img_arr: a numpy-like array
                 dimension order: Z C Y X  (full form TZCYXS)
        img_meta may contain
            img_meta['imagej']['voxel_size_um']
            img_meta['oblique_image']
    Return:
        vtkImageImport() object
    """
    # Ref:
    # Numpy 3D array into VTK data types for volume rendering?
    # https://discourse.vtk.org/t/numpy-3d-array-into-vtk-data-types-for-volume-rendering/3455/2
    # VTK Reading 8-bit tiff files (solved) VTK4.2
    # https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

    # code from Slicer
    # https://github.com/Slicer/Slicer/blob/2515768aaf70c161d781ff41f36f2a0655c88efb/Base/Python/slicer/util.py#L1950
    # def updateVolumeFromArray(volumeNode, img_arr):

    # See https://python.hotexamples.com/examples/vtk/-/vtkImageImport/python-vtkimageimport-function-examples.html

    dbg_print(4, 'ImportImageArray(): importing image of size:',  img_arr.shape)

    # Wild guess number of channels
    if len(img_arr.shape) == 4:
        n_ch = img_arr.shape[1]
    else:
        n_ch = 1

    if (img_meta is not None) and ('imagej' in img_meta) and \
       (img_meta['imagej'] is not None):
        if 'voxel_size_um' in img_meta['imagej']:
            if isinstance(img_meta['imagej']['voxel_size_um'], str):
                voxel_size_um = img_meta['imagej']['voxel_size_um'][1:-1]
                voxel_size_um = tuple(map(float, voxel_size_um.split(',')))
            else:  # assume array
                voxel_size_um = img_meta['imagej']['voxel_size_um']
        elif ('spacing' in img_meta['imagej']) and \
             ('XResolution' in img_meta) and \
             ('YResolution' in img_meta):
            voxel_size_um = (
                img_meta['XResolution'][0] / img_meta['XResolution'][1], \
                img_meta['YResolution'][0] / img_meta['YResolution'][1], \
                img_meta['imagej']['spacing'])
        else:
            voxel_size_um = (1.0, 1.0, 1.0)
    else:
        voxel_size_um = (1.0, 1.0, 1.0)

    img_importer = vtkImageImport()
    # Note: if img_arr is contiguous, 'simg is img_arr' is True
    simg = np.ascontiguousarray(img_arr, img_arr.dtype)  # maybe .flatten()?
    img_importer.CopyImportVoidPointer(simg.data, simg.nbytes)
    # To use SetImportVoidPointer, we need to keep a reference to simg some 
    #  where, to avoid GC and eventually Segmentation fault.
    #img_importer.SetImportVoidPointer(simg.data)
    if img_arr.dtype == np.uint8:
        img_importer.SetDataScalarTypeToUnsignedChar()
    elif img_arr.dtype == np.uint16:
        img_importer.SetDataScalarTypeToUnsignedShort()
    else:
        raise 'Unsupported format'
    img_importer.SetNumberOfScalarComponents(n_ch)
    img_importer.SetDataExtent (0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    img_importer.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)

    # the 3x3 matrix to rotate the coordinates from index space (ijk) to physical space (xyz)
    b_oblique_correction = img_meta.get('oblique_image', False)
    dbg_print(4, 'voxel_size_um       : ', voxel_size_um)
    dbg_print(4, 'b_oblique_correction: ', b_oblique_correction)
    if b_oblique_correction:
        img_importer.SetDataSpacing(voxel_size_um[0], voxel_size_um[1],
                                    voxel_size_um[2]*sqrt(2))
        rotMat = [ \
            1.0, 0.0,            0.0,
            0.0, cos(45/180*pi), 0.0,
            0.0,-sin(45/180*pi), 1.0
        ]
        img_importer.SetDataDirection(rotMat)
    else:
        img_importer.SetDataSpacing(voxel_size_um)

    return img_importer

def ImportImageFile(file_name, extra_conf = None):
    """
    Import image to vtkImageImport() to have a connection.
    Input:
        file_name: may be .tif .h5 .ims
        extra_conf: additional metadata, 
            typically image spacing specification.
            the extra_conf takes higher priority than meta data in the file
    Return:
        a vtkImageImport() object.
    """
    img_arr, img_meta = Read3DImageDataFromFile(file_name, extra_conf)
    if extra_conf:
        img_meta.update(extra_conf)
    img_import = ImportImageArray(img_arr, img_meta)
    return img_import

class OnDemandVolumeLoader():
    """
    Load image blocks upon request.
    Request parameters are a position and a radius.
    All image blocks intersect with the sphere will be loaded.
    
    TODO: add off-load.
    """
    def __init__(self):
        self.vol_list = []
        self.vol_origin = np.zeros((0,3), dtype=np.float64)
        self.vol_size   = np.zeros((0,3), dtype=np.float64)
    
    def ImportLychnixVolume(self, vol_list_file):
        from os.path import dirname, join, normpath
        jn = json.loads(open(vol_list_file).read())
        base_dir = normpath(join(dirname(vol_list_file), jn['image_path']))
        dbg_print(4,  'ImportLychnixVolume():')
        dbg_print(4,  '  voxel_size:', jn['voxel_size'])
        dbg_print(4,  '  channels  :', jn['channels'])
        dbg_print(4,  '  base_dir  :', base_dir)
        self.ImportVolumeList(jn['images'], basedir=base_dir)

    def ImportVolumeList(self, vol_list, basedir=''):
        from os.path import dirname, join, normpath
        # format of vol_list:
        # vol_list = [
        #   {
        #       "image_path": "full/path/to/tiff",
        #       "origin": [x, y, z],
        #       "size": [i, j, k]
        #   },
        #   ...
        # ]
        ap_list = [
            {
                'image_path': normpath(join(basedir, it['image_path'])),
                'origin': str2array(it['origin']),
                'size': str2array(it['size'])
            }
            for it in vol_list
        ]
        self.vol_list += ap_list
        self.vol_origin = np.concatenate(
            (
                self.vol_origin,
                _a([it['origin'] for it in ap_list])
            ), axis = 0)
        self.vol_size = np.concatenate(
            (
                self.vol_size,
                _a([it['size'] for it in ap_list])
            ), axis = 0)
#        print(self.vol_list)
#        print(self.vol_origin)
#        print(self.vol_size)
        
    def LoadVolumeAt(self, pos, radius=0):
        pos = _a([[pos[0], pos[1], pos[2]]])
        vol_center = self.vol_origin + self.vol_size / 2
        distance = np.abs(vol_center - pos)
        idx_in_range = np.flatnonzero(
            (distance[:,0] <= self.vol_size[:,0]/2 + radius) &
            (distance[:,1] <= self.vol_size[:,1]/2 + radius) &
            (distance[:,2] <= self.vol_size[:,2]/2 + radius) )
#        print(idx_in_range)
#        print('pos', pos)
#        print('origin:', self.vol_origin[idx_in_range, :])
#        print('size  :', self.vol_size  [idx_in_range, :])
        selected_vol = [self.vol_list[it] for it in idx_in_range]
        dbg_print(5, 'Volume(s) to be loaded: ', selected_vol)
        return selected_vol

def LoadSWCTree(filepath):
    """
    Load SWC file, i.e. tracing result.
    Return tree data structure.
    Tree data structure:
      (
        [(id, parent_id, type), ...],
        [(x, y, z, diameter), ...]
      )
    """
    d = np.loadtxt(filepath)
    if len(d) == 0:
        raise IndexError('Empty SWC file?')
    elif len(d.shape) == 1:
        d = d[np.newaxis, :]
    tr = (dtype_id(d[:,np.array([0,6,1])]),
          dtype_coor(d[:, 2:6]))

    # Treat id==parent as root node (a very special convention)
    tr[0][tr[0][:,0] == tr[0][:,1], 1] = -1

    # checking number of roots
    id_root = np.flatnonzero(tr[0][:,1] <= -1)
    n_tree = len(id_root)
    if n_tree > 1:
        swc_file = os.path.basename(filepath)
        dbg_print(3, 'LoadSWCTree(): Multiple roots detected:', swc_file)
        for j in range(n_tree):
            # output each root node
            nid = id_root[j]
            s2 = array2str(tr[1][nid], sep="")
            dbg_print(3, f'    : root {j+1} : {tr[0][nid]}, {s2}')

    # check uniqueness of node ID
    q, cnt = np.unique(tr[0][:,0], return_counts=True)
    idx_repeat = np.flatnonzero(cnt > 1)
    if len(idx_repeat) > 0:
        swc_file = os.path.basename(filepath)
        dbg_print(2, 'LoadSWCTree(): repeated node id detected:', swc_file)
        for j, idx_r in enumerate(idx_repeat):
            dbg_print(2, f'    : {j+1}. repeated id: {q[idx_r]}')

    #if not np.all(tr[0][1:,0] >= tr[0][:-1,0]):
    #    dbg_print(3, 'SplitSWCTree(): Node id not sorted.')
    return tr

def SWCNodeRelabel(tr, output_map = False):
    """
    Re-label node id in tr, s.t. node id = index i.e. 0, 1, 2, ...
    """
    tr_idx = tr[0].copy()
    max_id = max(tr_idx[:,0])   # max occur node index
    n_id = tr_idx.shape[0]      # number of nodes
    # relabel array (TODO: if max_id >> n_id, we need a different algo.)
    # map: id -> consecutive numbers (index)
    map_id_idx = np.zeros(max_id + 2, dtype = dtype_id)
    map_id_idx[-1] = -1
    #map_id_idx = -1 * np.ones(max_id + 2, dtype = dtype_id)
    map_id_idx[tr_idx[:, 0]] = np.arange(n_id, dtype=dtype_id)
    tr_idx[:, 0:2] = map_id_idx[tr_idx[:, 0:2]]
    if output_map:
        return tr_idx, map_id_idx
    else:
        return tr_idx

def SplitSWCTree(tr):
    """
    Split the tree(s) `tr` into linear segments, i.e. processes.
    Input : a swc tree ([(id0, pid0, ..), (id1, pid1, ..), ...], [..])
            not modified.
    Return: processes in index of tr. [[p0_idx0, p0_idx1, ...], [p1...]]
            Note that idx# is the index of tr, not the index(id) in tr.
    (No more: Assume tr is well and sorted and contains only one tree.)
    Assume root node id is '-1'.
    Multiple trees allowed.
    Usage example:
        tr = LoadSWCTree(name)
        processes = SplitSWCTree(tr)
    """
    tr_idx = SWCNodeRelabel(tr)
    n_id = tr_idx.shape[0]      # number of nodes
    # find branch/leaf points
    n_child,_ = np.histogram(tr_idx[:, 1],
                    bins = np.arange(-1, n_id + 1, dtype = dtype_id))
    n_tree = n_child[0]
    # leave out the node '-1'
    n_child = np.array(n_child[1:], dtype=dtype_id)
    # n_child == 0: leaf
    # n_child == 1: middle of a path or root
    # n_child >= 2: branch point
    id_bounds = np.nonzero(n_child - 1)[0]
    processes = []
    for eid in id_bounds:
        # travel from leaf to branching point or root
        i = eid
        filament = array.array(np_typecodes_map[dtype_id], [i])
        i = tr_idx[i, 1]  # parent
        while n_child[i] == 1 and i != -1:
            filament.append(i)
            i = tr_idx[i, 1]  # parent
        if i != -1:
            filament.append(i)
        elif len(filament) == 1:  # soma
            continue
        processes.append(np.array(filament[::-1], dtype=dtype_id))

    return processes

def SWCDFSOrder(processes):
    """
    Used with 
        SplitSWCTree()
        s.processes
    # test:
    s = swcs['201']
    processes = s.processes
    """
    if len(processes) == 0:
        return np.zeros((0,), dtype = dtype_id)
    # Get simplified tree nodes: [(branchleaf_id, parent_id), ...]
    tr_branch = np.array([(p[-1], p[0]) for p in processes], dtype = dtype_id)
    # find roots, i.e. nodes with non-exist parents
    root_ids = np.setdiff1d(tr_branch[:,1], tr_branch[:,0])

    # find index of root
    sid = np.argsort(tr_branch[:,1])
    id_p = np.searchsorted(np.sort(tr_branch[:,1]), root_ids)
    id_p = sid[id_p]
    idx_start = np.ones(len(processes), dtype = dtype_id)
    idx_start[id_p] = 0
    #assert s.tree_swc[0][processes[id_p[-1]][0], 1] == -1

    od_idx = np.hstack([p[idx_start[j]:] for j, p in enumerate(processes)])
    #assert len(np.unique(od_idx)) == len(od_idx)
    #assert max(od_idx)+1 == len(od_idx)

    # occationally, there is root point of no branches at all,
    # thus cannot captured by processes
    singleton = np.setdiff1d(np.arange(max(od_idx)+1), od_idx, True)
    od_idx = np.hstack((singleton, od_idx))

    return od_idx

def SWCDFSSort(tr, processes):
    idx = SWCDFSOrder(processes)
    idx_rev = np.zeros(idx.shape, dtype=idx.dtype)
    idx_rev[idx] = np.arange(len(idx))
    tr = (tr[0][idx], tr[1][idx])
    processes = [idx_rev[p] for p in processes]
    return tr, processes

def TreeNodeInfo(tr, node_idx):
    # assume no loop
    tr_rel = SWCNodeRelabel(tr)
    i = node_idx
    dist = 0
    ancestors = []
    while i != -1:
        ancestors.append(i)
        p = tr_rel[i,1]
        if p != -1:
            dist += np.linalg.norm(tr[1][p] - tr[1][i])
        i = p

    ancestors = np.array(ancestors, dtype = dtype_id)

    n_id = tr_rel.shape[0]      # number of nodes
    n_child,_ = np.histogram(tr_rel[:, 1],
                    bins = np.arange(-1, n_id + 1, dtype = dtype_id))
    n_child = np.array(n_child[1:], dtype=dtype_id)
    n_child[tr_rel[:, 1] == -1] = 2

    info = {
        #'ancestors': ancestors,
        'node_depth': len(ancestors) - 1,
        'branch_depth': np.sum(n_child[ancestors[1:]] - 1 > 0),
        'root_distance': dist
    }
    return info

def SimplifyTreeWithDepth(processes, output_mode = 'simple'):
    """
    Construct simplified tree from neuronal processes.
    Usage example:
        ps = SplitSWCTree(swcs[0].swc_tree)
        u = SimplifyTreeWithDepth(ps)
    Causion: singleton node will not be repesented in the final results.
    """
    n_branch = len(processes)
    # Get simplified tree nodes: [(branchleaf_id, parent_id), ...]
    tr_branch = np.array([(p[-1], p[0]) for p in processes], dtype = dtype_id)
    # find roots, i.e. nodes with non-exist parents
    root_ids = np.setdiff1d(tr_branch[:,1], tr_branch[:,0])
    n_root = len(root_ids)
    # init a tree: tr_simple = [(id, pid), ... ]
    # fill roots and branches
    tr_simple = np.zeros((n_branch + n_root, 2), dtype = dtype_id)
    tr_simple[:n_root, 0] = root_ids
    tr_simple[:n_root, 1] = -1
    tr_simple[n_root:, :] = tr_branch
    # map from id to index
    map_idx = dict(zip(tr_simple[:,0], range(tr_simple.shape[0])))
    # compute depth by traverse
    depth = np.zeros(n_branch + n_root, dtype = dtype_id)
    for j in range(n_root, n_branch + n_root):
        depth[j] = depth[map_idx[tr_simple[j,1]]] + 1
    
    if output_mode == 'simple':
        # [(node_id, parent_id, depth(root=0)), ...]
        u = np.hstack((tr_simple, depth[:,np.newaxis]))
    elif output_mode == 'depth':
        u = depth[n_root:,np.newaxis]
    return u

def GetUndirectedGraph(tr):
    """ return undirected graph of the tree, root (-1) is stripped. """
    tr_idx = SWCNodeRelabel(tr)
    tr_idx = np.array(tr_idx)
    n_id = tr_idx.shape[0]      # number of nodes

    # remove edge s.t. parent = -1
    negative_parent_idx, = np.nonzero(tr_idx[:, 1] == -1)
    bidx = np.ones(tr_idx.shape[0], dtype=bool)
    bidx[negative_parent_idx] = False
    tr_idx = tr_idx[bidx, :]
    if len(negative_parent_idx) > 1:
        dbg_print(4, 'GetUndirectedGraph(): multiple roots! ', negative_parent_idx)

    # Generate undirected graph
    ij = np.concatenate([tr_idx[1:, 0:2], tr_idx[1:, 1::-1]], axis=0)
    kk = np.ones(ij.shape[0], dtype = np.int8)
    coo = scipy.sparse.coo_matrix((kk, (ij[:,0], ij[:,1])),
                                  shape=(n_id, n_id))
    graph = scipy.sparse.csr_matrix(coo)
#    to get edges of node k: graph[k].indices

    return graph

