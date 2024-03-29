A python program (scripts) for viewing and inspecting of volumetric data and neuron morphology data using [VTK](https://vtk.org/) with 3D computer graphics experience.

Supported data format:

* [TIFF](https://docs.openmicroscopy.org/ome-model/6.1.1/ome-tiff/) and [IMS (HDF5)](https://imaris.oxinst.com/support/imaris-file-format) for volumetric data.
* [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) for neuron morphology data.

Typical use cases:

* Viewing the data with your favorite angle or perspective.
* Load and inspect SWC files, relevant information will be shown for clicked node.
* Run custom scripts (plugin) to analyze or alter the data, then seeing the result right away.
* Press F2 to go to command line mode for prototyping, i.e. type `swcs.line_width = 2` to set line width of all SWCs.
* Work with the source code for your own craft.

Features

* Thanks to VTK, we can also view in stereo modes (binocular 3D), such as split the view port horizontal, red blue or interlaced.
* Parallel loading of SWC files for speeding up work flow.
* Example plugins for coloring SWC files, e.g. color by node depth.
* Measure distance between selected points.
* Facilities for easier work with SWCs, such as batch read / assignment.
* Save the current scene (including view angle) to a file.
* The plugin system can essentially let you do almost any modification to the program.
* Customizable key bindings, through plugins/plugin.conf.json.

Demo

![Rendering Mouselight data](/examples/screenshot_mouse_light.png)


# Installation

## Install Python3

* Linux

    It is most likely already installed.

* Windows

    Try follow the [official directive](https://www.python.org/downloads/windows/). Install the latest stable version is recommended (at the time of writing, it is 3.11.1). For greater compatibility and stability, try stable version of last minor version (such as 3.10.9).

## Install dependencies

* Use pip

    `pip install vtk tifffile h5py scipy IPython joblib`

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

# Developer notes

## Reload a modified module during running

Method 1 example:

```python
    import sys
    import importlib
    importlib.reload(sys.modules['neu3dviewer.data_loader'])
    from neu3dviewer.data_loader import *
    #now neu3dviewer.data_loader get refreshed
```

Method 2 example (need to run in ipython):

```python
    # Ref. https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
    # at the start of a working session. (now this step is automatically done when pressed F2)
    %reload_ext autoreload
    %autoreload 2

    # then work as usual, the imported objects will be updated
    from neu3dviewer.data_loader import *
    help(Struct)
    # modify class Struct, then it is different
    help(Struct)
```

