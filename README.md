# FedCod: An Efficient Communication Protocol for Cross-Silo Federated Learning with Coding

This repository contains the Official Implementation for FedCod: An Efficient Communication Protocol for Cross-Silo Federated Learning with Coding.


## Configuration
We have written an example configuration file `cfg.ini`, which is used for FL training ResNet152 on CIFAR10 dataset, and communication with 9 clients in one GPU cluster. If you want to conduct an experiment in different GPU clusters, please re-write the IP addresses, ports and other parameters according to your requirements.

## Data Preparation
Split the CIFAR10 dataset into $n$ parts for $n$ clients by execute `$ python data_distribution.py`. The datasets for local training are located at `distributed_dataset/data/`.

## Example: Communication in Local GPU Cluster
Script `bash.sh` can start the server process and $9$ client processes to test training and communication within a GPU cluster. 

## Communication through WAN
After updating the configuration file, you can start FL training by running `python server.py --config_section server --commu fedcod` on the server and running `python client.py --config_section client{$i} --commu fedcod` on the client-$i. We recommend using *ansible* to manage multiple GPU clusters.


```
default: baseline, server-client communication
ncdown: Only apply network coding in download phase
ncddown: Only apply network coding with one-hop direct forwarding in download phase
ncupload: Only apply network coding in upload phase
nc: network coding in both download and upload phases
ncd: network coding with one-hop direct forwarding in both download and upload phases
agr: pre-aggregation in non-waiting mode for upload phase
agrw: pre-aggregation in waiting mode for upload phase
ncagr: network coding for both phases, and pre-aggregation in non-waiting mode for upload phase
ncagrw: network coding for both phases, and pre-aggregation in waiting mode for upload phase
ncdagrw: network coding with one-hop direct forwarding for both phases, and pre-aggregation in waiting mode for upload phase
dncagr:dynamic redundancy, network coding for both phases, and pre-aggregation in waiting mode for upload phase
dncdagr: dynamic redundancy, network coding with one-hop direct forwarding for both phases, and pre-aggregation in waiting mode for upload phase
hier: hierachical FL
```

## Requirements
Some required python libraries:
```
numpy
websockets
torch
torchvision
grpcio
protobuf
multiprocess
pandas
scikit-learn
```