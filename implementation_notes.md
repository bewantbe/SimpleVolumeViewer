# Implementation Notes

## Known bugs

### Double WindowResizeEvent
On Windows, when switch between non-full screen and full screen, we could receive double Resize Event.

### The embeded IPython shell is buggy
The embeded IPython shell either by `IPython.terminal.embed` or `IPython.start_ipython`
are not so robust. `embed()` cannot accept complex nested syntax, `start_ipython` sometimes
enters unrecoverable state that keep shouting errors.

### Buggy double click event
The double click event in VTK is very buggy, so we implement it ourself.

### VTK is not HiDPI display friendly
There is no easy way to get the true DPI of the display in VTK. The 
`DetectDPI` and `GetDPI` in `class vtkWindow` simply return wrong result (e.g. 96), 
what's worse, `DetectDPI` will silently change DPI setting of Text. The solution will
be setting the display compatibility in Windows.

Other solution is to use Qt, which is not desired (breaks the rule of simplicity).

## Performance tip
(very old version)
For memory footprint:
n_neuron = 1660 (SWC), n_points = 39382068 (0.44 GiB)
float32 mode:
RAM = 2.4GiB (5.3GiB during pick), 3.2g(after pick),
GPU = 1128MiB
float64 mode:
RAM = 3.3GiB (8.3GiB during pick), 3.6g(after pick),
GPU = 1128MiB

6357 swcs
```
# xle Linux, python 3.9.2 + numpy 1.24.2 + vtk 9.2.2
t_load = 136.121~150 sec
t_vtk_take = 54.435 sec
RAM  usage: 43.5GB
after a pick 47.9GB
GRAM usage: 6453MB

# xle Win11, python 3.9.11 + numpy 1.24.1 + vtk 9.2.5
t_load = 89.892 sec
t_vtk_take = 68.341 sec
RAM  usage: 31.0GB(1/1.024)
after a pick 34.95GB(1/1.024)
GRAM usage: 6.4GB
```

## Generate a standalone executable application

Use `pyinstaller` with `joblib` is not possible due to the bug:
    https://github.com/joblib/joblib/issues/1002

The workaround parallel_backend("multiprocessing") do not work due to 
   `AttributeError: Can't pickle local object 'GUIControl.AddBatchSWC.<locals>.batch_load'`

Seems there is solution for `joblib` padding: https://github.com/joblib/loky/pull/375

Use of [`nuitka`](https://nuitka.net/doc/user-manual.html) instead of `pyinstaller` has a similar problem:
    https://github.com/Nuitka/Nuitka/issues/2035

Use of `multiprocessing` instead of `joblib` has a similar problem:
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
But it is solvable (move the parallel object to global to avoid pickle problem), and it is currently the only solution.

```
# Fix WARNING for nuitka
 pip install --force-reinstall pywin32

# Failed Fix for pyinstaller
import joblib
#from multiprocessing import freeze_support
#from joblib import parallel_backend
#parallel_backend("multiprocessing")
#if __name__ == '__main__':
#    # Pyinstaller fix
#    freeze_support()
#    main()
```
