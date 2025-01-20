'''
Coding Client:
Download: (k,r) RS-code and direct forward, without ACK
Upload: (k,r) RS-code and direct forward (upload first, forward second)
'''

import multiprocessing as mp
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
from .base_client import Client

class CodClient(Client):
    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.address_dict['server'] = [self.args.server_ip, self.args.server_port]
        conf = ConfigParser()
        conf.read('cfg.ini')
        for i in range(self.args.num_users):
            if i == self.args.idx_users:
                continue
            client_name = 'client' + str(i)
            client_port = conf.get(client_name, 'client_port')
            client_ip = conf.get(client_name, 'client_ip')
            self.address_dict[client_name] = [client_ip, client_port]
        self.iter = 0
        self.ID = self.args.idx_users
        
    def encoder(self, params, coding):
        arr_local = params.reshape(-1)
        model_local = coding.encode_RS(arr_local, self.args.upload_k, self.args.upload_r)
        return model_local

    def decoding(self, part_list, order_list, coding):        
        model_glob = np.array([])
        for i in order_list:
            model_glob = np.append(model_glob, part_list[i])
        model_glob = model_glob.reshape(self.args.download_k, -1)
        model_glob = coding.decode_RS(model_glob, self.args.download_k, self.args.download_r, order_list)
        # reshape
        model_glob = model_glob.reshape(-1)

        return model_glob


    def decoder(self, receive_queue, coding, local_iter, log):
        part_idx_list = []
        part_list = [None] * (self.args.download_k + self.args.download_r)
        model_glob = None
        
        while True:
            shm_name, rec_time = receive_queue.get()
            data = self.pop_shared_data(shm_name)
            if data == b'000':
                os.kill(os.getppid(), signal.SIGTERM)
                return None
            
            if int(data[3:6]) != local_iter:
                continue

            if data[:3] == b'999':
                send_data = b'998' + data[3:]
                shm_name = self.push_shared_data(send_data)
                self.send_queue.put((shm_name, 'all'))
            
            part_idx = int(data[6:9])
            log.info('Iteration '+ str(local_iter) + ". Block " + str(part_idx) + " is received at {0} ".format(rec_time))

            if part_idx not in part_idx_list:
                part_glob = pickle.loads(data[9:])
                part_list[part_idx] = part_glob
                part_idx_list.append(part_idx)
            if len(part_idx_list) == self.args.download_k:
                pre_code_time = time.time()
                log.info('Iteration '+ str(self.iter.value) + ". Server blocks is received at {0} ".format(pre_code_time))
                model_glob = self.decoding(part_list.copy(), part_idx_list.copy(), coding)
                cur_code_time = time.time()
                log.info('Iteration '+ str(self.iter.value) + "。 Decoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))
                break

        return model_glob

    def upload(self, model_local):
        for idx, part in enumerate(model_local):
            model_local_byte = pickle.dumps(model_local[idx])
            user_str = str(self.args.idx_users)  # change user_idx to str
            user_str = user_str.zfill(3)  # change user_idx into 3 digit
            idx_str = str(idx)  # change idx to str
            idx_str = idx_str.zfill(3)  # change idx into 3 digit
            iter_str = str(self.iter.value)  # change iter to str
            iter_str = iter_str.zfill(3)  # change iter into 3 digit
            model_local_byte = b'997' + bytes(iter_str, encoding="utf8") + bytes(idx_str, encoding="utf8") + bytes(user_str, encoding="utf8") + model_local_byte
            
            shm_name = self.push_shared_data(model_local_byte)
            if idx < self.args.upload_k:
                self.send_queue.put((shm_name, 'server'))
            elif idx%(self.args.num_users-1) < self.ID:
                self.send_queue.put((shm_name, 'client'+str(idx%(self.args.num_users-1))))
            elif idx%(self.args.num_users-1) >= self.ID:
                self.send_queue.put((shm_name, 'client'+str(idx%(self.args.num_users-1)+1)))
                            
    def MsgStream(self, request_iterator, context):
        data_chunks = []

        for data_chunk in request_iterator:
            data_chunks.append(data_chunk.message)


        data = b''.join(data_chunks)
        data_time = time.time()

        # iter+1 data can only come from the global model download process
        # if it emerges, refresh the queues 
        if int(data[3:6])-1 == self.iter.value:
            self.iter.value += 1

        if data[:3] == b'999':
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name, data_time))           
            # download relay
            # send_data = b'998' + data[3:]
            # self.send_queue.put((send_data, 'all'))
        elif data[:3] == b'998':
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name, data_time))
        elif data[:3] == b'997':
            # upload relay
            send_data = b'996' + data[3:]
            shm_name = self.push_shared_data(send_data)
            self.buffer_queue.put((shm_name))
        elif data[:3] == b'000':
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name, data_time))

        return communication_pb2.response(message=b'ok')
    

    def processor(self):
        log = create_logger(self.loggername)
        log.info("Training")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model().to(device)
        datas, labels = self.load_dataset()
        self.train_data = DataLoader(TensorDataset(datas.to(device),labels.to(device)), self.args.local_bs,shuffle=True,drop_last=False)
        # coding = Coding()
        coding = OptimizedCoding()
        local_iter = -1
        while True:
            local_iter += 1
            model_glob = self.decoder(self.receive_queue, coding, local_iter, log)
            cur_train_time = time.time()
            log.info('Iteration '+ str(self.iter.value) + ". Training starts at {0} ".format(cur_train_time))
            if model_glob is None:
                return
            
            model_dict = rebuilt_dict_flatten(model_glob, self.model.state_dict())
            self.model.load_state_dict(deepcopy(model_dict))
            # self.model.to(device)
            self.model.train()
            log.info(str(self.iter.value)+" Training start.")
            self.train_iter(log)
            # self.model.cpu()
            new_model_dict = self.model.state_dict()
            updates = get_updates_flatten(new_model_dict, model_dict)
            pre_code_time = time.time()
            model_local = self.encoder(updates, coding)
            cur_code_time = time.time()
            log.info('Iteration '+ str(self.iter.value) + ". Local Encoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))
            cur_up_time = time.time()
            log.info('Iteration '+ str(self.iter.value) + ". Uploading starts in {0} ".format(cur_up_time))
            self.upload(model_local)

    async def parallel_sender(self, iter, key, queue, logo):
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        log = create_logger(self.loggername)
        channel = grpc.insecure_channel(uri)

        while True:
            shm_name = queue.get()
            data = self.pop_shared_data(shm_name)

            iter_ster = int(data[3:6])
            part_idx = int(data[6:9])
            cur_time = time.time()
            if data[:3] == b'998':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " at {0} ".format(cur_time))
            elif data[:3] == b'997':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " at {0} ".format(cur_time))
            elif data[:3] == b'996':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " at {0} ".format(cur_time))
            log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
            
            if int(data[3:6]) == iter.value:
                # await self.send(data, uri)
                await self.lc_send(data, channel)

            cur_time = time.time()
            if data[0:3] == b'998':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " Over! at {0} ".format(cur_time))
            elif data[0:3] == b'997':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " Over! at {0} ".format(cur_time))
            elif data[0:3] == b'996':
                log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " Over! at {0} ".format(cur_time))
     
    def upload_sender_func(self, iter, key, queue, buffer_queue, log):
        asyncio.run(self.upload_sender(iter, key, queue, buffer_queue, log))

    async def upload_sender(self, iter, key, queue, buffer_queue, logo):
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        log = create_logger(self.loggername)
        channel = grpc.insecure_channel(uri)

        while True:
            if not queue.empty():
                shm_name = queue.get()
                data = self.pop_shared_data(shm_name)

                iter_ster = int(data[3:6])
                part_idx = int(data[6:9])
                cur_time = time.time()
                if data[:3] == b'998':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " at {0} ".format(cur_time))
                elif data[:3] == b'997':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " at {0} ".format(cur_time))
                elif data[:3] == b'996':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " at {0} ".format(cur_time))
                log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
                
                if int(data[3:6]) == iter.value:
                    # await self.send(data, uri)
                    await self.lc_send(data, channel)

                cur_time = time.time()
                if data[0:3] == b'998':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " Over! at {0} ".format(cur_time))
                elif data[0:3] == b'997':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " Over! at {0} ".format(cur_time))
                elif data[0:3] == b'996':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " Over! at {0} ".format(cur_time))
            elif not buffer_queue.empty():
                shm_name = buffer_queue.get()
                data = self.pop_shared_data(shm_name)

                iter_ster = int(data[3:6])
                part_idx = int(data[6:9])

                if int(data[3:6]) != iter.value:
                    continue

                cur_time = time.time()
                if data[:3] == b'998':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " at {0} ".format(cur_time))
                elif data[:3] == b'997':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " at {0} ".format(cur_time))
                elif data[:3] == b'996':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " at {0} ".format(cur_time))
                log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
                
                await self.send(data, uri)

                cur_time = time.time()
                if data[0:3] == b'998':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is sent to " + key + " Over! at {0} ".format(cur_time))
                elif data[0:3] == b'997':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is upload to " + key + " Over! at {0} ".format(cur_time))
                elif data[0:3] == b'996':
                    log.info("Iteration " + str(iter_ster) + ", Block " + str(part_idx) + " is fwd to " + key + " Over! at {0} ".format(cur_time))
            

    def parallel_sender_func(self, iter, key, queue, log):
        asyncio.run(self.parallel_sender(iter, key, queue, log))


    async def parallel_senders(self):     
        log = create_logger(self.loggername)
        await asyncio.sleep(5)

        # global model data relay
        queue_list = {}
        for key in self.address_dict.keys():
            queue_list[key] = mp.Queue()

        process_list = []
        for key in self.address_dict.keys():
            if key != 'server':
                p = mp.Process(target=self.parallel_sender_func, args=(self.iter, key, queue_list[key], log))
            else:
                p = mp.Process(target=self.upload_sender_func, args=(self.iter, key, queue_list[key], self.buffer_queue, log))
            p.daemon = True
            p.start()
            process_list.append(p)

        while True:
            shm_name, target = self.send_queue.get()
            data = self.pop_shared_data(shm_name)
            if data == b'000':
                os.kill(os.getppid(), signal.SIGTERM)
                return
            if int(data[3:6]) != self.iter.value:
                continue
            if data[0:3] == b'998':
                for key in self.address_dict.keys():
                    if key != 'server':
                        shm_name = self.push_shared_data(data)
                        queue_list[key].put(shm_name)
            elif data[0:3] == b'997':
                shm_name = self.push_shared_data(data)
                queue_list[target].put(shm_name)
            elif data[0:3] == b'996':
                shm_name = self.push_shared_data(data)
                queue_list['server'].put(shm_name)
                    
        for p in process_list:
            p.join()
      # print("Queue join.")
        for key in queue_list.keys():
            queue_list[key].join()
            
      # print("Over")

    def sender(self):        
        asyncio.run(self.parallel_senders())

    async def manager(self, log):
        self.send_queue = mp.Queue()
        self.receive_queue = mp.Queue()
        self.buffer_queue = mp.Queue()
        process_list = []
        self.iter = mp.Value('i', 0)

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
     
