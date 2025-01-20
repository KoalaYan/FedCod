'''
Coding Server:
Download: (k,r) RS-code and direct forward, without ACK
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


class CodServer(Server):

    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.iter = 0

    def encoder(self, params, coding):
        arr_glob = params.reshape(-1)
        model_glob = coding.encode_RS(arr_glob, self.args.download_k, self.args.download_r)
        return model_glob

    def decoding(self, client_idx, part_list, order_list, coding):        
        model_local = np.array([])
        for i in order_list:
                model_local = np.append(model_local,
                                       part_list[client_idx][i])
        model_local = model_local.reshape(self.args.upload_k, -1)
        model_local = coding.decode_RS(model_local, self.args.upload_k, self.args.upload_r,
                                      order_list)
        # reshape
        model_local = model_local.reshape(-1)

        return model_local

    # upload: 997 + user indx + iteration number + data partition index + data
    def decoder(self, receive_queue, coding, log):
        part_idx_list, part_list = structure(self.args)
        param_list = []
        client_num_wait = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        while True:
            shm_name, data_time = receive_queue.get()
            data = self.pop_shared_data(shm_name)
            if int(data[3:6]) == self.iter:
                part_idx = int(data[6:9])
                client_idx = int(data[9:12])
                log.info('Iteration '+ str(self.iter) + ". Client " + str(client_idx) + " block " + str(part_idx) + " is received at {0} ".format(data_time))  
            
                if part_idx in part_idx_list[client_idx]:
                    continue
                else:
                    part_local = pickle.loads(data[12:])
                    part_list[client_idx][part_idx] = part_local
                    part_idx_list[client_idx].append(part_idx)
                if len(part_idx_list[client_idx]) == self.args.upload_k:
                    pre_code_time = time.time()
                    log.info('Iteration '+ str(self.iter) + ". Client " + str(client_idx) + " blocks is received at {0} ".format(pre_code_time))
                    local_model = self.decoding(client_idx, part_list.copy(), part_idx_list[client_idx].copy(), coding)
                    cur_code_time = time.time()
                    log.info('Iteration '+ str(self.iter) + ". Client " + str(client_idx) + "Decoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))
                    param_list.append(torch.tensor(local_model).reshape(-1,1).to(device))
                    client_num_wait += 1
            if client_num_wait == self.args.num_users:
                break
        return param_list

    # download: 999 + iteration number + data partition index + data
    def distribute(self, send_queue, model_glob, iter):
        for idx, part in enumerate(model_glob):
            model_glob_byte = pickle.dumps(part)

            idx_str = str(idx)  # change idx to str
            idx_str = idx_str.zfill(3)  # change idx into 3 digit
            iter_str = str(iter)  # change iter to str
            iter_str = iter_str.zfill(3)  # change iter into 3 digit
            send_byte = b'999' + bytes(iter_str, encoding="utf8") + bytes(idx_str, encoding="utf8") + model_glob_byte
            key = 'client' + str(idx % self.args.num_users)
            shm_name = self.push_shared_data(send_byte)
            send_queue.put((shm_name, key))
    
    async def async_processor(self):
        log = create_logger(self.loggername)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pretrain_model_dict = torch.load('./resnet152_cifar10_2.pth')
        self.model = self.create_model().to(device)
        self.model.train()
        # self.model.load_state_dict(pretrain_model_dict)
        self.dataset = self.load_dataset()
        # coding = Coding()
        coding = OptimizedCoding()
        last_time = time.time()
        for t in range(self.args.epochs):
            self.iter = t
            params  = get_params_flatten(self.model.state_dict())
            pre_code_time = time.time()
            model_glob = self.encoder(params, coding)
            cur_code_time = time.time()
            log.info('Iteration '+ str(t) + ". Encoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))

            cur_time = time.time()
            log.info('Iteration '+ str(t) + ". Operation took around {0} seconds to complete".format(cur_time - last_time))
            last_time = cur_time
            log.info('Iteration '+ str(t) + ". Distribution starts at {0}".format(cur_time))

            self.distribute(self.send_queue, model_glob, t)
            param_list = self.decoder(self.receive_queue, coding, log)
            self.glob_iter.value += 1
            
            cur_up_time = time.time()
            log.info('Iteration '+ str(t) + ". Uploading ends at {0} ".format(cur_up_time))
            model_dict = algorithms.fedavg(param_list, self.model, self.args)
            self.model.load_state_dict(model_dict)
            correct = self.evaluate()
            log.info(
                'Round {:3d}, Test accuracy: {:.2f}'.format(
                    t, correct))
        for key in self.address_dict.keys():
            print(key, " Shutdown.")
            self.send_queue.put((b'000', key))
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
            
            iter_idx = int(data[3:6])
            if iter_idx != self.glob_iter.value:
                continue
            part_idx = int(data[6:9])
            cur_time = time.time()
            log.info("Block " + str(part_idx) + " is sent to " + key + " at {0} ".format(cur_time))
            log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
            # await self.send(data, uri)
            await self.lc_send(data, channel)
            cur_time = time.time()
            log.info("Block " + str(part_idx) + " is sent to " + key + " Over! at {0} ".format(cur_time))

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
            if int(data[3:6]) != self.glob_iter.value:
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
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name,cur_time))
        # else:
        #     print('wrong!!!', int(data[3:6], self.glob_iter.value))
        return communication_pb2.response(message=b'ok')
        
    async def manager(self, log):
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
