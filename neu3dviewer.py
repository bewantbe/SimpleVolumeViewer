#!/usr/bin/env python3
# make this a package:
# pyinstaller --collect-all vtkmodules .\neu3dviewer.py
# now see dist: .\dist\neu3dviewer\neu3dviewer.exe
# Ref.
#   https://pyinstaller.org/en/stable/usage.html
#   https://realpython.com/pyinstaller-python/

# temp run
#from os.path import realpath, dirname
#import sys
#file_path = realpath(__file__)
#sys.path.insert(0, dirname(file_path))

from neu3dviewer.img_block_viewer import main

main()