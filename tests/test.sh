#!/bin/bash
# run test in this directory
# must run on Linux with X

ref_png_dir='ref_screenshot/'
test_png_dir=''

prog='/usr/bin/env python3 ../img_block_viewer.py --verbosity 2 --off_screen_rendering 1'

echo "Test --help"
$prog --help
if [ $? != 0 ] ; then
  echo "!!! Error"
fi

echo "Test 1"
$prog --filepath ./ref_data/RM06_s56_c10_f3597_p0.tif --colorscale 4 --save_screen "${test_png_dir}test.png"
cmp "${test_png_dir}test.png" "${ref_png_dir}1.png"
rm "${test_png_dir}test.png"

if [ $? != 0 ] ; then
  echo "!!! Error"
else
  echo "    Passed"
fi
