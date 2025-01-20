import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import websockets.client
import websockets.server
import asyncio
import time
import torch
from configparser import ConfigParser
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import functools
import pickle
from copy import deepcopy
import os
import signal
from threading import Thread
from queue import Queue
import heapq
import sys
# import psutil

from utils.config_to_arg import argument
from utils.logger import create_logger # Logger
from utils.coding import Coding, Ratelesscoding, OptimizedCoding, NetworkCoding
from utils.get_params import rebuild_dict, rebuilt_dict_flatten, get_params, get_updates, get_updates_flatten, get_updates_flatten_network_test
import models
from node import Node

from concurrent import futures
import grpc
from ..protocol import communication_pb2, communication_pb2_grpc

class HierClient(Node, communication_pb2_grpc.StreamServicer):
    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.address_dict = {}
        self.address_dict['hierserver'] = [self.args.hierserver_ip, self.args.hierserver_port]

    def load_dataset(self):
        datas = torch.load('./distributed_dataset/data/'+str(self.args.idx_users)+'.pt')
        labels = torch.load('./distributed_dataset/label/'+str(self.args.idx_users)+'.pt')
        return datas, labels
   
    def train_iter(self, log):
        net = self.model
        optimizer = self.optimizer

        for iter in range(self.args.local_ep):
            for j, item in enumerate(self.train_data):
                x, y = item
                optimizer.zero_grad()
                output = net(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                # utils.clip_gradient(optimizer=optimizer, grad_clip=1e-2)
                optimizer.step()
        
        # correct = 0
        # with torch.no_grad():
        #     net.eval()
        #     for j, item in enumerate(self.train_data):
        #         data, target = item
        #         output = net(data)
        #         pred = output.argmax(dim=1, keepdim=True)
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        # correct = 1. * correct / (len(self.train_data) * self.args.local_bs)
        # net.train()
        # log.info('Test accuracy: {:.2f}'.format(correct))

        return net.state_dict()
    

    def processor(self):
        log = create_logger(self.loggername)
        log.info("Training")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        datas, labels = self.load_dataset()
        dataset_size = len(datas)
        self.train_data = DataLoader(TensorDataset(datas.to(device),labels.to(device)), self.args.local_bs,shuffle=True,drop_last=False)
        iter = 0
        while True:
            shm_name, rec_time = self.receive_queue.get()
            data = self.pop_shared_data(shm_name)
            if data == b'000':
                break
            model_glob = pickle.loads(data[3:])

            log.info('Iteration '+ str(iter) + ". Block is received at {0} ".format(rec_time))
            cur_train_time = time.time()
            log.info('Iteration '+ str(iter) + ". Training starts at {0} ".format(cur_train_time))            
            model_dict = rebuilt_dict_flatten(model_glob, self.model.state_dict())
            self.model.load_state_dict(deepcopy(model_dict))
            # self.model.to(device)
            self.model.train()
            log.info("Training start.")
            self.train_iter(log)
            log.info("Training end.")
            # self.model.cpu()
            new_model_state = self.model.state_dict()
            params = get_updates_flatten(new_model_state, model_dict)
            params = params.reshape(-1)
            cur_up_time = time.time()
            log.info('Iteration '+ str(iter) + ". Uploading starts in {0} ".format(cur_up_time))
            model_local_byte = pickle.dumps(params)
            user_str = str(self.args.idx_users)  # change user_idx to str
            user_str = user_str.zfill(3)  # change user_idx into 3 digit
            shm_data = b'997' + bytes(user_str, encoding="utf8") + model_local_byte

            shm_name = self.push_shared_data(shm_data)
            self.send_queue.put(shm_name)
            iter += 1
        os.kill(os.getppid(), signal.SIGTERM)

    
    def sender(self):
        asyncio.run(self.single_sender())

    async def single_sender(self):        
        log = create_logger(self.loggername)
        await asyncio.sleep(5)
        key = 'hierserver'
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        channel = grpc.insecure_channel(uri)

        while True:
            shm_name = self.send_queue.get()
            data = self.pop_shared_data(shm_name)

            cur_time = time.time()
            log.info("Block is sent to " + key + " at {0} ".format(cur_time))
            log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
            # await self.send(data, uri)
            await self.lc_send(data, channel)
            cur_time = time.time()
            log.info("Block is sent to " + key + " Over! at {0} ".format(cur_time))