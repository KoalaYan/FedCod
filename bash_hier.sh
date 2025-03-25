#!/bin/bash
gpu=(5 5 4 4 3 3)
cfg=(1 1 2 2 3 3)
cfg_list=(1 2 3)
idx_list=(0 1 0 1 0 1)
export CUDA_VISIBLE_DEVICES=7
python server.py --config_section server --commu hier --cfg_fn hierfl/server&  # default nc ncd ncagr ncagrw ncdagrw dncagr

for i in {0..2};do
export CUDA_VISIBLE_DEVICES=6
python client.py --config_section hierserver --commu hier --cfg_fn hierfl/hier_${cfg_list[i]}&
done

for i in {0..5};do
export CUDA_VISIBLE_DEVICES=${gpu[i]}
python client.py --config_section client${idx_list[i]} --commu hier --cfg_fn hierfl/hier_${cfg[i]}& 
done
# python server.py --config_section server --commu lc_default 
# python client.py --config_section client0 --commu default
# python client.py --config_section client1 --commu default
# python client.py --config_section client2 --commu default