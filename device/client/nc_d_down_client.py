'''
Network Coding Client:
Download: random linear network coding and forward, with ACK
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


class NCDDownClient(Client):
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
                        
    def decoder(self, nc, local_iter, log):
        idx_matrix = None
        part_list = []
        block_list = []
        model_glob = None
        
        while True:
            shm_name, rec_time = self.receive_queue.get()
            data = self.pop_shared_data(shm_name)
            if data == b'000':
                os.kill(os.getppid(), signal.SIGTERM)
                return None
            
            if int(data[3:6]) != local_iter:
                continue
            
            # if data[:3] == b'999':
            #     send_data = b'998' + data[3:]
            #     self.send_queue.put((send_data, 'all'))

            log.info('Iteration '+ str(local_iter) + ". Block is received at {0} ".format(rec_time))
            block = pickle.loads(data[6:])
            block = block.reshape(-1)
            index = block[:self.args.download_k]
            data_block = block[self.args.download_k:]
            if idx_matrix is not None:
                tmp_matrix = np.vstack((idx_matrix, index))
                rank_matrix = np.linalg.matrix_rank(tmp_matrix)
                if rank_matrix != len(block_list) + 1:
                    continue
            else:
                tmp_matrix = index
                rank_matrix = 1

            idx_matrix = tmp_matrix
            part_list.append(data_block)
            block_list.append(block)
            log.info("Recent block number is " + str(len(block_list)) + ".")

            # encoded_block = nc.encoding(block_list, 0, 128, 1)            
            # model_glob_byte = pickle.dumps(encoded_block)
            # iter_str = str(local_iter)  # change iter to str
            # iter_str = iter_str.zfill(3)  # change iter into 3 digit
            # send_data = b'998' + bytes(iter_str, encoding="utf8") + model_glob_byte
            # send_queue.put((send_data, 'all'))
            # time.sleep(0.1)
            

            if len(part_list) == self.args.download_k:
                self.ack()
                pre_code_time = time.time()
                log.info('Iteration '+ str(self.iter.value) + ". Server blocks is received at {0} ".format(pre_code_time))
                decoded_blocks = nc.decoding(part_list.copy(), idx_matrix.copy())
                model_glob = decoded_blocks.reshape(-1)
                cur_code_time = time.time()
                log.info('Iteration '+ str(self.iter.value) + "。 Decoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))
                break

        return model_glob
    
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
            self.flag.value = 0
            for i in range(len(self.status_table)):
                self.status_table[i] = 0

        if data[:3] == b'999':# or data[:3] == b'998':
            if self.flag.value == 0:        
                # download relay
                send_data = b'998' + data[3:]
                shm_name = self.push_shared_data(send_data)
                self.send_queue.put((shm_name, 'all'))

                shm_name = self.push_shared_data(data)
                self.receive_queue.put((shm_name, data_time))   
        elif data[:3] == b'998':
            if self.flag.value == 0:
                shm_name = self.push_shared_data(data)
                self.receive_queue.put((shm_name, data_time))
        elif data[:3] == b'666':
            client_id = int(data[6:9])
            self.status_table[client_id] = 1

        elif data[:3] == b'000':
            shm_name = self.push_shared_data(data)
            self.receive_queue.put((shm_name, data_time))

        return communication_pb2.response(message=b'ok')

    # 666 [0:3] + iteration number [3:6] + user index [6:9]
    def ack(self):
        user_str = str(self.args.idx_users)  # change user_idx to str
        user_str = user_str.zfill(3)  # change user_idx into 3 digit
        iter_str = str(self.iter.value)  # change iter to str
        iter_str = iter_str.zfill(3)  # change iter into 3 digit
        msg_byte = b'666' + bytes(iter_str, encoding="utf8") + bytes(user_str, encoding="utf8")
        print('ACK')
        shm_name = self.push_shared_data(msg_byte)
        self.send_queue.put((shm_name, 'all'))
    
    def processor(self):
        log = create_logger(self.loggername)
        log.info("Training")
        log.info("Worker Process ID:"+str(mp.current_process().pid))
        # log.info("Before Setting Affinity:"+str(psutil.Process().cpu_affinity()))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model().to(device)
        datas, labels = self.load_dataset()
        self.train_data = DataLoader(TensorDataset(datas.to(device),labels.to(device)), self.args.local_bs,shuffle=True,drop_last=False)
        # coding = Coding()
        nc = NetworkCoding(self.args.download_k)
        coding = OptimizedCoding()
        local_iter = -1
        while True:
            local_iter += 1
            model_glob = self.decoder(nc, local_iter, log)
            self.flag.value = 1
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
            new_model_state = self.model.state_dict()
            params = get_updates_flatten(new_model_state, model_dict)
            params = params.reshape(-1)
            cur_up_time = time.time()
            log.info('Iteration '+ str(self.iter.value) + ". Uploading starts in {0} ".format(cur_up_time))
            model_local_byte = pickle.dumps(params)
            user_str = str(self.args.idx_users)  # change user_idx to str
            user_str = user_str.zfill(3)  # change user_idx into 3 digit
            iter_str = str(self.iter.value)  # change iter to str
            iter_str = iter_str.zfill(3)  # change iter into 3 digit
            shm_data = b'997' + bytes(iter_str, encoding="utf8") + bytes(user_str, encoding="utf8") + model_local_byte

            shm_name = self.push_shared_data(shm_data)
            self.send_queue.put((shm_name, 'server'))

    def parallel_sender_func(self, iter, key, queue, ack, log):
        asyncio.run(self.parallel_sender(iter, key, queue, ack, log))

    async def parallel_sender(self, iter, key, queue, ack, logo):
        uri = f"{self.address_dict[key][0]}:{self.address_dict[key][1]}"
        log = create_logger(self.loggername)
        channel = grpc.insecure_channel(uri)

        while True:
            if ack == None or ack.empty():
                if not queue.empty():
                    shm_name = queue.get()
                    data = self.pop_shared_data(shm_name)

                    iter_ster = int(data[3:6])
                    # if data[:3] == b'998' and self.status_table[int(key[6:])] != 0:
                    #     log.info(key + " not relay.")
                    #     continue
                    cur_time = time.time()
                    if data[:3] == b'998':
                        log.info("Iteration " + str(iter_ster) + ", Block is sent to " + key + " at {0} ".format(cur_time))
                    elif data[:3] == b'997':
                        log.info("Iteration " + str(iter_ster) + ", Block is upload to " + key + " at {0} ".format(cur_time))
                    log.info("Block size is {0} ".format(data.__sizeof__()/(2**20)))
                    
                    if int(data[3:6]) == iter.value:
                        # await self.send(data, uri)
                        await self.lc_send(data, channel)

                    cur_time = time.time()
                    if data[0:3] == b'998':
                        log.info("Iteration " + str(iter_ster) + ", Block is sent to " + key + " Over! at {0} ".format(cur_time))
                    elif data[0:3] == b'997':
                        log.info("Iteration " + str(iter_ster) + ", Block is upload to " + key + " Over! at {0} ".format(cur_time))
            else:
                shm_name = ack.get()
                data = self.pop_shared_data(shm_name)
                if int(data[3:6]) == iter.value:
                    # await self.send(data, uri)
                    await self.lc_send(data, channel)
    
    async def parallel_senders(self):
        log = create_logger(self.loggername)
        await asyncio.sleep(3)

        # global model data relay
        queue_list = {}
        for key in self.address_dict.keys():
            queue_list[key] = mp.Queue()

        # send acknowledgement to neighbours and shutdown signal
        ack_queue_list = {}
        for key in self.address_dict.keys():
            ack_queue_list[key] = mp.Queue()

        process_list = []
        for key in self.address_dict.keys():
            p = mp.Process(target=self.parallel_sender_func, args=(self.iter, key, queue_list[key], ack_queue_list[key], log))
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
                    if key == 'server':
                        continue
                    if self.status_table[int(key[6:])] == 0:
                        shm_name = self.push_shared_data(data)
                        queue_list[key].put(shm_name)
            elif data[0:3] == b'997':
                shm_name = self.push_shared_data(data)
                queue_list[target].put(shm_name)
            elif data[0:3] == b'996':
                shm_name = self.push_shared_data(data)
                queue_list['server'].put(shm_name)
            elif data[:3] == b'666':
                for key in self.address_dict.keys():
                    shm_name = self.push_shared_data(data)
                    ack_queue_list[key].put(shm_name)
                    
        for t in process_list:
            t.join()
      # print("Queue join.")
        for key in queue_list.keys():
            queue_list[key].join()
        for key in self.address_dict.keys():
            ack_queue_list[key].join()
            
      # print("Over")

    def sender(self):        
        asyncio.run(self.parallel_senders())

    async def manager(self, log):
        self.status_table = mp.Array('i', [0] * self.args.num_users)
        self.send_queue = mp.Queue()
        self.receive_queue = mp.Queue()
        process_list = []
        self.iter = mp.Value('i', 0)
        self.flag = mp.Value('i', 0)

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