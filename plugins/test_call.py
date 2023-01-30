#

import numpy as np
import tifffile

from neu3dviewer.data_loader import (
    Read3DImageDataFromFile,
    Save3DImageToFile,
)

def PluginMain(ren, iren, gui_ctrl):
    file_name = "tests/ref_data/RM06_s56_c10_f3597_p0.tif"
    img_arr, img_meta = Read3DImageDataFromFile(file_name)
    Save3DImageToFile('c.tif', img_arr, img_meta)
    gui_ctrl.RemoveObject('volume2')
    gui_ctrl.AddObjectProperty('volume2',
        {
            "type": "volume",
            "opacity_transfer_function": {
                "AddPoint": [
                    [20, 0.1],
                    [255, 0.9]
                ],
                "opacity_scale": 1.0
            },
            "color_transfer_function": {
                "AddRGBPoint": [
                    [0.0, 0.0, 0.0, 0.0],
                    [64.0, 0.2, 0.2, 0.0],
                    [128.0, 0.5, 0.5, 0.0],
                    [255.0, 0.9, 0.9, 0.0]
                ],
                "trans_scale": 1.0
            },
            "interpolation": "cubic"
        } )
    gui_ctrl.AddObject('volume2', {
        "type"           : "volume",
        "property"       : "volume2",
        "origin"         : [100, 100, 0],
        "rotation_matrix": [1,0,0, 0,1,0, 0,0,1],
        "file_path"      : file_name,
    })
    iren.GetRenderWindow().Render()
