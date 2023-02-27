#!/bin/bash
# run test in this directory
# must run on Linux with X

# exit on non-zero return value
set -e

prog='../neu3dviewer.py --window_size 800x600 --no_interaction 1 --off_screen_rendering 1 --verbosity 2'

ref_pic_dir='ref_screenshot/'
test_pic_dir='./tmpdir'
mkdir -p "$test_pic_dir"

echo "Test --help"
$prog --help

echo "Test 1"
$prog --screenshot "$test_pic_dir/1.png" --img_path ref_data/RM06_s56_c10_f3597_p0.tif --colorscale 3
cmp "$ref_pic_dir/1.png" "$test_pic_dir/1.png"

#for ((i=0; i<6; i++))
#do
#  xdotool search --name Neuron3DViewer key minus
#done
#xdotool search --name Neuron3DViewer key 'p'
#xdotool search --name Neuron3DViewer key 'q'

echo "Test 2"
$prog --screenshot "$test_pic_dir/2.png" --swc_dir ./ref_data/swc_ext
cmp "$ref_pic_dir/2.png" "$test_pic_dir/2.png"

echo "Test 3"
$prog --screenshot "$test_pic_dir/3.png" --img_path ../twiddling/3810_48244_24534.h5 --range [0:64,:,:] --origin [-64,0,0] --colorscale 4
cmp "$ref_pic_dir/3.png" "$test_pic_dir/3.png"

echo "Test 4"
$prog --screenshot "$test_pic_dir/4.png" --scene ../scene_files/img_saved.json
cmp "$ref_pic_dir/4.png" "$test_pic_dir/4.png"

echo "Test 5"
$prog --screenshot "$test_pic_dir/5.png" --lychnis_blocks /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/image/blocks.json --colorscale 3.3 --swc /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/F5.json.swc --fibercolor yellow
cmp "$ref_pic_dir/5.png" "$test_pic_dir/5.png"

echo "Passed all."
