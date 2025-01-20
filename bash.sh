#!/bin/bash
gpu=(2 3 3 4 4 5 5 6 6 7 7)
export CUDA_VISIBLE_DEVICES=2
python server.py --config_section server --commu default&  # default fedcod cod ncagrw
for i in {0..8};do
export CUDA_VISIBLE_DEVICES=${gpu[i]}
python client.py --config_section client$i --commu default& 
done