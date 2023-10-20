# encoding: utf-8
# SPDX-License-Identifier: GPL-3.0-or-later

# Ref.
# Python package
# https://docs.python.org/3/library/__main__.html#main-py-in-python-packages
# Python Application
# https://docs.python.org/3/using/windows.html#python-application

from multiprocessing import Pool
from multiprocessing import freeze_support
from .img_block_viewer import main

if __name__ == '__main__':
    #freeze_support()
    main()