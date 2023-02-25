#!/usr/bin/env python


# Numpy 3D array into VTK data types for volume rendering?
# https://discourse.vtk.org/t/numpy-3d-array-into-vtk-data-types-for-volume-rendering/3455/2

# code from Slicer
# https://github.com/Slicer/Slicer/blob/2515768aaf70c161d781ff41f36f2a0655c88efb/Base/Python/slicer/util.py#L1950
# def updateVolumeFromArray(volumeNode, img_arr):

# help from https://public.kitware.com/pipermail/vtkusers/2003-November/020884.html

# python vtk_ex_raycast.py '/media/xyy/DATA/VISoRData/RM06_set2020-09-19/RM06_s128_c13_f8906_p3.tif'

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
# noinspection PyUnresolvedReferences
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper
from vtk import vtkTIFFReader, vtkImageReader
import vtk

import numpy as np
import tifffile

# copy from volumeio.py
# Read tiff file, return images and meta data
def read_tiff(tif_path, as_np_array = True):
    # see also https://pypi.org/project/tifffile/
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
    return images, metadata

# Read tiff file, return images and meta data
def read_tiff_meta(tif_path):
    # see also https://pypi.org/project/tifffile/
    tif = tifffile.TiffFile(tif_path)
    metadata = {tag_name:tag_val.value 
                for tag_name, tag_val in tif.pages[0].tags.items()}
    if hasattr(tif, 'imagej_metadata'):
        metadata['imagej'] = tif.imagej_metadata
    return metadata

def main():
    fileName = get_program_parameters()

    colors = vtkNamedColors()

    # This is a simple volume rendering example that
    # uses a vtkFixedPointVolumeRayCastMapper

    # Create the standard renderer, render window
    # and interactor.
    ren1 = vtkRenderer()

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren1)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data.
    # old TIFF reader
    #reader = vtkTIFFReader()
    #reader.SetFileName(fileName)
    #reader.Update()
    
    # read tiff
    #img_arr, img_meta = read_tiff(fileName)
    #
    #vshape      = tuple(reversed(img_arr.shape))
    #vcomponents = vshape[0]
    #vshape      = vshape[1:4]
    #import vtk.util.numpy_support
    #vtype = vtk.util.numpy_support.get_vtk_array_type(img_arr.dtype)

    #vimage = vtk.vtkImageData()
    #vimage.SetDimensions(vshape)
    #vimage.AllocateScalars(vtype, vcomponents)
    
    #
    # vtkAlgorithmOutput * vtkAlgorithm::GetOutputPort ( )
    
    
    #internal_arr = vtk.util.numpy_support.vtk_to_numpy(vimage.GetPointData().GetScalars()).reshape(nshape)
    
    #internal_arr[:,:,:] = img_arr
    # spacing
    #vimage.SetSpacing(1.0, 1.0, 2.5)
    #vimage.SetSetOrigin(0.0, 0.0, 0.0)
    #setDirectionMarix()
    #GetIndexToPhysicalMatrix()

    ## use image import
    # See https://python.hotexamples.com/examples/vtk/-/vtkImageImport/python-vtkimageimport-function-examples.html
    img_arr, img_meta = read_tiff(fileName)
    print(img_meta)
    n_ch = 1

    dataImporter = vtk.vtkImageImport()
    simg = np.ascontiguousarray(img_arr, img_arr.dtype)  # maybe .flatten()
    # see also: SetImportVoidPointer
    dataImporter.CopyImportVoidPointer(simg.data, simg.nbytes)
    if img_arr.dtype == np.uint8:
        dataImporter.SetDataScalarTypeToUnsignedChar()
    elif img_arr.dtype == np.uint16:
        dataImporter.SetDataScalarTypeToUnsignedShort()
    else:
        raise "Unsupported format"
    dataImporter.SetNumberOfScalarComponents(n_ch)
    dataImporter.SetDataExtent (0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    dataImporter.SetWholeExtent(0, simg.shape[2]-1, 0, simg.shape[1]-1, 0, simg.shape[0]-1)
    #dataImporter.SetDataSpacing(1.0, 1.0, 2.5)
    #dataImporter.setDataOrigin()
    #dataImporter.setDataDirection()
    #

    print(dataImporter.GetDataExtent())
    print(dataImporter.GetWholeExtent())

    # Create transfer mapping scalar value to opacity.
    opacity_scale = 40.0
    opacityTransferFunction = vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(opacity_scale*20, 0.0)
    opacityTransferFunction.AddPoint(opacity_scale*255, 0.2)

    # Create transfer mapping scalar value to color.
    trans_scale = 40.0
    colorTransferFunction = vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(trans_scale*0.0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*64.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*128.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(trans_scale*192.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(trans_scale*255.0, 0.0, 0.2, 0.0)

    # The property describes how the data will look.
    volumeProperty = vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    #volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.SetInterpolationType(vtk.VTK_CUBIC_INTERPOLATION)

    # The mapper / ray cast function know how to render the data.
    volumeMapper = vtkFixedPointVolumeRayCastMapper()
    # volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    #volumeMapper.SetBlendModeToComposite()
    #volumeMapper.SetInputConnection(reader.GetOutputPort())
    # vtkVolumeMapper
    # https://vtk.org/doc/nightly/html/classvtkVolumeMapper.html
    # SetInputConnection( vtkAlgorithmOutput *  input )
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume.
    volume = vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren1.AddVolume(volume)
    ren1.SetBackground(colors.GetColor3d('Wheat'))
    ren1.GetActiveCamera().Azimuth(45)
    ren1.GetActiveCamera().Elevation(30)
    ren1.ResetCameraClippingRange()
    ren1.ResetCamera()

    renWin.SetSize(2400, 1800)
    renWin.SetWindowName('SimpleRayCast')
    renWin.Render()

    iren.Start()


def get_program_parameters():
    import argparse
    description = 'Volume rendering of a high potential iron protein.'
    epilogue = '''
    This is a simple volume rendering example that uses a vtkFixedPointVolumeRayCastMapper.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filename', help='ironProt.vtk', default='/media/xyy/DATA/VISoRData/RM06_set2020-09-19/RM06_s128_c13_f8906_p3.tif')
    args = parser.parse_args()
    return args.filename


if __name__ == '__main__':
    main()
