# Test script
# run in ./tests

# exit on non-zero return value
set -e

cmdst='../neu3dviewer.py --window_size 800x600 --no_interaction 1 --off_screen_rendering 1'
tmpdir='./tmpdir'
mkdir -p "$tmpdir"

$cmdst --screenshot "$tmpdir/1.png" --img_path ref_data/RM06_s56_c10_f3597_p0.tif --colorscale 3

#for ((i=0; i<6; i++))
#do
#  xdotool search --name Neuron3DViewer key minus
#done
#xdotool search --name Neuron3DViewer key 'p'
#xdotool search --name Neuron3DViewer key 'q'

$cmdst --screenshot "$tmpdir/2.png" --img_path ../twiddling/3810_48244_24534.h5 --range [0:64,:,:] --origin [-64,0,0] --colorscale 4

$cmdst --screenshot "$tmpdir/3.png" --lychnis_blocks /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/image/blocks.json --colorscale 3.3 --swc /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/F5.json.swc --fibercolor yellow

$cmdst --screenshot "$tmpdir/4.png" --swc_dir ./ref_data/swc_ext

$cmdst --screenshot "$tmpdir/5.png" --scene ../scene_files/img_saved.json
