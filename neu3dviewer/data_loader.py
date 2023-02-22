# SPDX-License-Identifier: GPL-3.0-or-later

# data loader that do not involve too much of VTK

import pprint
import json
import time

import numpy as np
from numpy import sqrt, sin, cos, tan, pi
from numpy import array as _a
dtype_id = np.int32

import scipy.sparse

import tifffile
import h5py

from .utils import (
    dbg_print,
    str2array,
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

def Read3DImageDataFromFile(file_name, *item, **keys):
    if file_name.endswith('.tif') or file_name.endswith('.tiff'):
        img_arr, img_meta = read_tiff(file_name)
    elif file_name.endswith('.ims') or file_name.endswith('.h5'):
        img_arr, img_meta = read_ims(file_name, *item, **keys)
    else:
        raise TypeError('file format not supported: ' + file_name)
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
                voxel_size_um = tuple(map(float, voxel_size_um.split(', ')))
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
    Load image blocks upon request. TODO: add off-load.
    Request parameters are a position and a radius.
    All image blocks intersect with the sphere will be loaded.
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
    tr = (dtype_id(d[:,np.array([0,6,1])]),
          np.float64(d[:, 2:6]))
    return tr

def SplitSWCTree(tr):
    """
    Split the tree in a swc into linear segments, i.e. processes.
    Input : a swc tree ([(id0, pid0, ..), (id1, pid1, ..), ...], [..])
            not modified.
    Return: processes in index of tr. [[p0_idx0, p0_idx1, ...], [p1...]]
            Note that idx# is the index of tr, not the index in tr.
    Assume tr is well and sorted and contains only one tree.
    Usage example:
        tr = LoadSWCTree(name)
        processes = SplitSWCTree(tr)
    """

    # re-label index in tr, s.t. root is 0 and all followings continued
    tr_idx = tr[0].copy()
    max_id = max(tr_idx[:,0])   # max occur node index
    n_id = tr_idx.shape[0]      # number of nodes
    # relabel array (TODO: if max_id >> n_id, we need a different algo.)
    arr_full = np.zeros(max_id+2, dtype=dtype_id)
    arr_full[-1] = -1
    arr_full[tr_idx[:,0]] = np.arange(n_id, dtype=dtype_id)
    tr_idx[:,0:2] = arr_full[tr_idx[:,0:2]]
    # find branch points
    n_child,_ = np.histogram(tr_idx[1:,1],
                    bins = np.arange(n_id + 1, dtype = dtype_id))
    n_child = np.array(n_child, dtype=dtype_id)
    # n_child == 0: leaf
    # n_child == 1: middle of a path or root
    # n_child >= 2: branch point
    id_bounds = np.nonzero(n_child-1)[0]
    processes = []
    for eid in id_bounds:
        # travel from leaf to branching point or root
        i = eid
        filament = [i]
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

def SimplifyTreeWithDepth(processes):
    """
    Run SplitSWCTree(tr) first
    ps = SplitSWCTree(swcs[0].swc_tree)
    tr_s, depth = SimplifyTreeWithDepth(ps)
    u = np.hstack((tr_s, depth[:,np.newaxis]))
    # importlib.reload(sys.modules['neu3dviewer.data_loader'])
    # from neu3dviewer.data_loader import *
    # call SimplifyTreeWithDepth
    """
    # construct simplified tree
    # tr_s = [(id, pid), ... ]
    n_branch = len(processes)
    tr_s = np.zeros((n_branch + 1, 2), dtype = dtype_id)
    tr_s[0, :] = (0, -1)
    for j, p in enumerate(processes):
        tr_s[j+1, :] = (p[-1], p[0])
    # map id to index
    map_idx = dict(zip(tr_s[:,0], np.arange(n_branch + 1)))
    # compute depth
    depth = np.zeros(n_branch + 1, dtype = dtype_id)
    for j in range(1, n_branch + 1):
        depth[j] = depth[map_idx[tr_s[j,1]]] + 1
    # [(node_id, parent_id, depth(root=0)), ...]
    u = np.hstack((tr_s, depth[:,np.newaxis]))
    return u

def GetUndirectedGraph(tr):
    """ return undirected graph of the tree, root (-1) is stripped. """
    # re-label index in tr, this part is the same as SplitSWCTree()
    tr_idx = tr[0].copy()
    max_id = max(tr_idx[:, 0])  # max occur node index
    n_id = tr_idx.shape[0]  # number of nodes
    # relabel array (TODO: if max_id >> n_id, we need a different algo.)
    arr_full = np.zeros(max_id + 2, dtype=dtype_id)
    arr_full[-1] = -1
    arr_full[tr_idx[:, 0]] = np.arange(n_id, dtype=dtype_id)
    tr_idx[:, 0:2] = arr_full[tr_idx[:, 0:2]]
    tr_idx = np.array(tr_idx)

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
                                  shape=(n_id,n_id))
    graph = scipy.sparse.csr_matrix(coo)
#    to get edges of node k: graph[k].indices

    return graph

