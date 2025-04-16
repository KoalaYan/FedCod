#!/bin/bash
gpu=(1 1 1 2 2 2 0 0 0)
export CUDA_VISIBLE_DEVICES=7
python server.py --config_section server --commu allreduce&  # default, ncdown, ncddown, ncupload, nc, ncd, agr, agrw, ncagr, ncagrw, ncdagrw, dncagr, dncdagr, allreduce
for i in {0..8};do
export CUDA_VISIBLE_DEVICES=${gpu[i]}
python client.py --config_section client$i --commu allreduce&
done