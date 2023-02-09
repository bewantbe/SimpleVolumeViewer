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

    `pip install vtk tifffile h5py scipy IPython`

    Or

    `pip install -r requirements.txt`

    or if you are in mainland China and need a faster pip download speed:

    `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

    It is recommended to use the latest stable version of vtk. e.g. vtk 8.2 has a known bug affects this program, and it is fixed in vtk 9.2.

    `pip install --upgrade vtk`

## Download code

* Use git

    `git clone <url>`

    For example `<url>` could be:

      https://github.com/bewantbe/SimpleVolumeViewer.git

* Or, download current version

    Click the "[Download ZIP](https://github.com/bewantbe/SimpleVolumeViewer/archive/refs/heads/main.zip)".

    Unzip it somewhere, there is no need to "install" it.

# Running

* In command line (terminal) window

    `cd` to where you put this repo, e.g.:

    `cd SimpleVolumeViewer`
    
    Run:

    `python3 -m neu3dviewer`

    then drag-and-drop file(s) to the GUI.
    
    Or, with options:

    `python3 -m neu3dviewer <options>`
    
    e.g. to load all SWCs in a directory:

    `python3 -m neu3dviewer --swc_dir brain_data\v1.6.2\swc_registered_low\`

    See `python3 -m neu3dviewer --help` for full list of options.

* Use the "Windows" way:

  Double click the executable `SimpleVolumeViewer/bin/neu3dviewer.bat` the drag-and-drop.

  Or create/use a shortcut like `SimpleVolumeViewer/bin/neu3dviewer_py310`. It is recommended to alter the compatibility option for HiDPI to fix the blur problem.
  e.g. in properties of the shortcut open Compatibility tab, -> Change high DPI settings -> Override high DPI scaling behavior. Scaling performed by: Application.

* To get help

    `python3 -m neu3dviewer --help`

    Or, press h key in the GUI.
