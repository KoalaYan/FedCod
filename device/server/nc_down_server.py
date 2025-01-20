'''
Network Coding Server:
Download: random linear network coding and forward, with ACK
Upload: (k,r) RS-code and direct forward (upload first, forward second)
'''

import multiprocessing as mp
import numpy as np
import websockets.client
import websockets.server
import asyncio
from configparser import ConfigParser
import functools
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision
import threading
import os
import signal
from threading import Thread
from queue import Queue
import time

from utils.config_to_arg import argument
from utils.logger import create_logger
from utils.get_params import get_params, get_params_flatten, rebuilt_dict_flatten
from utils.coding import structure, Coding, Ratelesscoding, OptimizedCoding, NetworkCoding
from data_distribution import HAR_dataloader
from node import Node
import algorithms
import models

from concurrent import futures
import grpc
from ..protocol import communication_pb2, communication_pb2_grpc
from .base_server import Server


class NCDownServer(Server):

    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.iter = 0

    def encoder(self, params, coding):
        arr_glob = params.reshape(-1)
        model_glob = coding.encode_RS(arr_glob, self.args.download_k, self.args.download_r)
        return model_glob
    
    # download: 999 + iteration number + data partition index + data
    def distribute(self, params, nc, iter, log):
        blocks = nc.split(params)
        # encoded_blocks = nc.encoding(blocks, 0, 128, self.args.download_k * self.args.num_users)
        for i in range(self.args.download_k):
            encoded_blocks = nc.encoding(blocks, 0, 128, self.args.num_users)
            for j in range(self.args.num_users):
                if self.status_table[j] != 0:
                    continue

                # encoded_block = nc.encoding(blocks, 0, 128, 1)
                encoded_block = encoded_blocks[j]
                
                model_glob_byte = pickle.dumps(encoded_block)

                iter_str = str(iter)  # change iter to str
                iter_str = iter_str.zfill(3)  # change iter into 3 digit
                send_byte = b'999' + bytes(iter_str, encoding="utf8") + model_glob_byte
                key = 'client' + str(j)
                shm_name = self.push_shared_data(send_byte)
                self.send_queue.put((shm_name, key))
                # log.info("Queue Block size is {0} ".format(send_byte.__sizeof__()/(2**20)))
                    
    async def async_processor(self):
        log = create_logger(self.loggername)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pretrain_model_dict = torch.load('./resnet152_cifar10_2.pth')
        self.model = self.create_model().to(device)
        self.model.train()
        # self.model.load_state_dict(pretrain_model_dict)
        self.dataset = self.load_dataset()
        # coding = Coding()
        nc = NetworkCoding(self.args.download_k)
        coding = OptimizedCoding()
        last_time = time.time()
        for t in range(self.args.epochs):
            self.iter = t
            params  = get_params_flatten(self.model.state_dict())
            cur_time = time.time()
            log.info('Iteration '+ str(t) + ". Operation took around {0} seconds to complete".format(cur_time - last_time))
            last_time = cur_time
            log.info('Iteration '+ str(t) + ". Distribution starts at {0}".format(cur_time))

            self.distribute(params, nc, t, log)
            count = 0
            param_list = []
            while count < self.args.num_users:
                shm_name, rec_time = self.receive_queue.get()
                data = self.pop_shared_data(shm_name)
                client_idx = int(data[6:9])
                log.info('Iteration '+ str(t) + ". Client " + str(client_idx) + " is received at {0} ".format(rec_time))
                count += 1
                param_list.append(pickle.loads(data[9:]))
                
            cur_up_time = time.time()
            log.info('Iteration '+ str(t) + ". Uploading ends at {0} ".format(cur_up_time))
            param_list = [torch.Tensor(x).reshape(-1,1).to(device) for x in param_list]
            cur_2_time = time.time()
            model_dict = algorithms.fedavg(param_list, self.model, self.args)
            cur_1_time = time.time()
            log.info('Iteration '+ str(t) + ". Aggregation took around {0} seconds to complete".format(cur_1_time - cur_2_time))
            self.model.load_state_dict(model_dict)

            for i in range(self.args.num_users):
                self.status_table[i] = 0
            
            cur_up_time = time.time()
            log.info('Iteration '+ str(t) + ". Uploading ends at {0} ".format(cur_up_time))
            model_dict = algorithms.fedavg(param_list, self.model, self.args)
            self.model.load_state_dict(model_dict)
            self.glob_iter.value += 1
            correct = self.evaluate()
            log.info(
                'Round {:3d}, Test accuracy: {:.2f}'.format(
                    t, correct))
            
        print(" Shutdown.")
        shm_name = self.push_shared_data(b'000')
        self.send_queue.put((shm_name, 'server'))
        time.sleep(5)
        print("Kill!")
        os.kill(os.getppid(), signal.SIGTERM)

    
    async def parallel_sender(self, key, queue, logo):
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        log = create_logger(self.loggername)
        channel = grpc.insecure_channel(uri)
        
        while True:
            shm_name = queue.get()
            data = self.pop_shared_data(shm_name)
            if int(data[3:6]) != self.glob_iter.value or self.status_table[int(key[6:])] != 0:
                continue
            iter_idx = int(data[3:6])
            cur_time = time.time()
            log.info("Block is sent to " + key + " at {0} ".format(cur_time))
            log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
            # await self.send(data, uri)
            await self.lc_send(data, channel)
            cur_time = time.time()
            log.info("Block is sent to " + key + " Over! at {0} ".format(cur_time))
    
    async def parallel_senders(self):
        await asyncio.sleep(3)
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
            # cur_time = time.time()
            # log.info("Get data from send queue at {0} ".format(cur_time))
            shm_name, target = self.send_queue.get()
            data = self.pop_shared_data(shm_name)
            # cur_time = time.time()
            # log.info("Gotten data from send queue at {0} ".format(cur_time))
            if data == b'000':
                # server shutdown
                for key in self.address_dict.keys():
                    shm_name = self.push_shared_data(data)
                    queue_list[key].put(shm_name)
                    websocket_num -= 1
                    if websocket_num == 0:
                        break
                    continue

            if int(data[3:6]) != self.glob_iter.value:
                continue
            if self.status_table[int(target[6:])] != 0:
                continue

            shm_name = self.push_shared_data(data)
            queue_list[target].put(shm_name)

    def MsgStream(self, request_iterator, context):
        data_chunks = []

        for data_chunk in request_iterator:
            data_chunks.append(data_chunk.message)

        data = b''.join(data_chunks)
        cur_time = time.time()

        # print(f"Received {len(data)} bytes of data.")
        if int(data[3:6]) == self.glob_iter.value:
            if data[:3] == b'666':
                client_id = int(data[6:9])
                self.status_table[client_id] = 1
            else:
                shm_name = self.push_shared_data(data)
                self.receive_queue.put((shm_name,cur_time))
            
        # else:
        #     print('wrong!!!', int(data[3:6], self.glob_iter.value))
        return communication_pb2.response(message=b'ok')
        
    async def manager(self, log):
        self.status_table = mp.Array('i', [0] * self.args.num_users)
        self.send_queue = mp.Queue()
        self.receive_queue = mp.Queue()
        process_list = []
        self.glob_iter = mp.Value('i', 0)
        
        processor = mp.Process(target=self.processor,args=(
        ))
        process_list.append(processor)
        processor.daemon = True
        processor.start()

        sender = mp.Process(target=self.sender,args=(
        ))
        process_list.append(sender)
        sender.daemon = False
        sender.start()
        log.info("Initialization done.")
                
        
        # Replace websockets.server.serve with gRPC server initialization
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        communication_pb2_grpc.add_StreamServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"{self.args.my_ip}:{self.args.my_port}")
        self.server.start()
        self.server.wait_for_termination()  # 阻塞调用线程，直到服务器终止
        print('Here')
