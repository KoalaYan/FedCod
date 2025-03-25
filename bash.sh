#!/bin/bash
gpu=(0 0 0 1 1 1 2 2 2)
export CUDA_VISIBLE_DEVICES=3
python server.py --config_section server --commu dncdagr&  # default, ncdown, ncddown, ncupload, nc, ncd, agr, agrw, ncagr, ncagrw, ncdagrw, dncagr, dncdagr
for i in {0..8};do
export CUDA_VISIBLE_DEVICES=${gpu[i]}
python client.py --config_section client$i --commu dncdagr&
done