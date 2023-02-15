# Test script

./img_block_viewer.py --filepath /home/xyy/code/py/vtk_test/RM06_s128_c13_f8906_p3.tif

for ((i=0; i<18; i++))
do
  xdotool search --name SimpleRayCast key minus
done

xdotool search --name SimpleRayCast key 'p'
xdotool search --name SimpleRayCast key 'q'

./img_block_viewer.py --filepath /home/xyy/code/py/vtk_test/3810_48244_24534.h5 --range [0:64,:,:] --origin [-64,0,0]

./img_block_viewer.py --lychnis_blocks /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/image/blocks.json --swc /media/xyy/DATA/RM006_related/big_traced/RM006-004-lychnis/F5.json.swc --fibercolor yellow

./img_block_viewer.py --swc_dir /home/xyy/Documents/SIAT_CAS/xu/tracing/swc_collect/v1.5_swc4web_20221125/swc_registered_low/

./img_block_viewer.py --scene img_saved.json
