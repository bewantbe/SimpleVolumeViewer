#

import numpy as np
import tifffile

def PluginMain(ren, iren, guictrl):
    file_name = 'b.tif'
    img_arr, img_meta = Read3DImageDataFromFile(file_name)
    Save3DImageToFile('c.tif', img_arr, img_meta)
    guictrl.RemoveObject('volume2')
    guictrl.AddObjectProperty('volume2',
        {
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
    guictrl.AddObject('volume2', {
        "type"           : "volume",
        "property"       : "volume2",
        "origin"         : [100, 100, 0],
        "rotation_matrix": [1,0,0, 0,1,0, 0,0,1],
        "file_path"      : "c.tif",
    })
    iren.GetRenderWindow().Render()
