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

class HierCenterClient(Node, communication_pb2_grpc.StreamServicer):
    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.address_dict = {}
        self.address_dict['server'] = [self.args.server_ip, self.args.server_port]
        for i in range(self.args.num_users):
            client_name = 'client' + str(i)
            self.address_dict[client_name] = [args.client_ip_list[i], args.client_port_list[i]]

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
    

    def MsgStream(self, request_iterator, context):
        data_chunks = []

        for data_chunk in request_iterator:
            data_chunks.append(data_chunk.message)

        data = b''.join(data_chunks)
        cur_time = time.time()

        if data[:3] == b'999':
            shm_name = self.push_shared_data(data)
            self.send_queue.put((shm_name,'all'))
            
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name,cur_time))
        elif data[:3] == b'997':
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name,cur_time))
            
        # else:
        #     print('wrong!!!', int(data[3:6], self.glob_iter.value))
        return communication_pb2.response(message=b'ok')
    
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

            count = 0
            param_list = [params]
            log.info('# of clients: '+str(self.args.num_users))
            while count < self.args.num_users:
                shm_name, rec_time = self.receive_queue.get()
                data = self.pop_shared_data(shm_name)
                client_idx = int(data[3:6])
                log.info('Iteration '+ str(iter) + ". Client " + str(client_idx) + " is received at {0} ".format(rec_time))
                count += 1
                param_list.append(pickle.loads(data[6:]))
                
            new_params = torch.mean(torch.tensor(np.array(param_list)), dim=0)
            cur_up_time = time.time()
            log.info('Iteration '+ str(iter) + ". Uploading starts in {0} ".format(cur_up_time))
            model_local_byte = pickle.dumps(new_params)
            user_str = str(self.args.cfg_fn[-1])  # change user_idx to str
            user_str = user_str.zfill(3)  # change user_idx into 3 digit
            shm_data = b'996' + bytes(user_str, encoding="utf8") + model_local_byte

            shm_name = self.push_shared_data(shm_data)
            self.send_queue.put((shm_name,'server'))
            iter += 1
        os.kill(os.getppid(), signal.SIGTERM)

    def sender(self):
        asyncio.run(self.parallel_senders())

    async def parallel_sender(self, key, queue, logo):
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        log = create_logger(self.loggername)
        channel = grpc.insecure_channel(uri)
        
        while True:
            shm_name = queue.get()
            data = self.pop_shared_data(shm_name)
        
            cur_time = time.time()
            log.info(f"Block is sent to {key} at {cur_time}")
            log.info(f"Block size is {data.__sizeof__() / (2**20)} MB")
            # await self.send(data, uri)
            await self.lc_send(data, channel)
            cur_time = time.time()
            log.info(f"Block is sent to {key} Over! at {cur_time}")

    def parallel_sender_func(self, key, queue, log):
        asyncio.run(self.parallel_sender(key, queue, log))

    async def parallel_senders(self):
        await asyncio.sleep(5)
        log = create_logger(self.loggername)


        procress_list = []
        queue_list = {}
        for key in self.address_dict.keys():
            queue_list[key] = mp.Queue()
            p = mp.Process(target=self.parallel_sender_func, args=(key, queue_list[key], log))
            p.daemon = True
            p.start()
            procress_list.append(p)
            
        websocket_num = len(procress_list)
        while True:
            shm_name, target = self.send_queue.get()
            data = self.pop_shared_data(shm_name)
            if data == b'000':
                # server shutdown
                for key in self.address_dict.keys():
                    shm_name = self.push_shared_data(data)
                    queue_list[key].put(shm_name)
                    websocket_num -= 1
                    if websocket_num == 0:
                        break
                    continue
                
            if target == 'all':
                for key in self.address_dict.keys():
                    if key == 'server':
                        continue
                    shm_name = self.push_shared_data(data)
                    queue_list[key].put(shm_name)
            elif target == 'server':
                shm_name = self.push_shared_data(data)
                queue_list[target].put(shm_name)
                


        for t in thread_list:
            t.join()
        for key in queue_list.keys():
            queue_list[key].join()

