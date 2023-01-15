A python program (scripts) for viewing and inspecting of volumetric data and neuron morphology data using [VTK](https://vtk.org/) with 3D computer graphics experience.

Supported data format:

* [TIFF](https://docs.openmicroscopy.org/ome-model/6.1.1/ome-tiff/) and [IMS (HDF5)](https://imaris.oxinst.com/support/imaris-file-format) for volumetric data.
* [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) for neuron morphology data.

Typical use cases:

* Viewing the data with you favorite angle or perspective.
* Run custom scripts (plugin) to analyze or alter the data, then seeing the result right away.
* Work with the source code for your own craft.

Features

* Thanks to VTK, we can also view in stereo modes (binocular 3D), such as split the view port horizontal, red blue or interlaced.
* Support scene file, which specifies the scene objects, make it easier for scripting.
* Save the current scene (including view angle) to a file.
* The plugin system can essentially let you do almost any modification to the program.

# Installation

## Install Python3

* Linux

    It is most likely already installed.

* Windows

    Try follow the [official directive](https://www.python.org/downloads/windows/). Install the latest stable version is recommended (at the time of writing, it is 3.11.1). For greater compatibility and stability, try stable version of last minor version (such as 3.10.9).

## Install dependencies

* Use pip

    `pip install vtk opencv-python tifffile h5py scipy joblib IPython`

# Running

* In command line (terminal) window

    `python3 img_block_viewer.py <options>`

* To get help

    `python3 img_block_viewer.py --help`

    Or, during running, press h key.

