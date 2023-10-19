# run in F2
from .data_loader import SplitSWCTree

# initialize

ren = gui_ctrl.GetMainRenderer()

# construct data from existing swc
nt       = swcs[1].tree_swc
nt[1][:,0:3] += 10*(np.random.rand(len(nt[1]), 3)-0.5)  # random shift
swc_conf = {
            "type": "swc",
            "file_path": "nothisfile.swc",
            "color": "green",
            "line_width": 2.0,
           }
name = 'a3ra'

# construct a swc
swco = gui_ctrl.translator.obj_swc(gui_ctrl, ren)
# processes, raw_points, ntree
swco.cache1 = SplitSWCTree(nt), nt[1][:,0:3], nt
sco = swco.parse(swc_conf)

# house-hold jobs of adding an object
gui_ctrl.point_set_holder.AddPoints(sco.PopRawPoints(), name)
gui_ctrl.scene_objects.update({name: sco})
gui_ctrl.scene_saved['objects'].update({name: swc_conf})


if 0:
    swcs.line_width = 2

if 0:
    gui_ctrl.RemoveObject(name)


