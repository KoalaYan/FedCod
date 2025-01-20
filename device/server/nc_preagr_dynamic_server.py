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


class NCAGRServer(Server):

    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.iter = 0
        self.upload_r = args.upload_r
        self.bottom = 0
        self.update_flag = False
        self.update_p = 0

    def encoder(self, params, coding):
        arr_glob = params.reshape(-1)
        model_glob = coding.encode_RS(arr_glob, self.args.download_k, self.args.download_r)
        return model_glob

    '''
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
    '''

    def decoding(self, part_list, order_list, coding):        
        model_local = np.array([])
        for i in order_list:
                model_local = np.append(model_local,
                                       part_list[i])
        model_local = model_local.reshape(self.args.upload_k, -1)
        model_local = coding.decode_RS(model_local, self.args.upload_k, self.args.upload_r,
                                      order_list)
        # reshape
        model_local = model_local.reshape(-1)

        return model_local
    
    # upload: 997 + iteration number + data partition index + client IO vector + data
    def decoder(self, coding, log):
        args = self.args

        part_idx_list = []
        part_list = [None] * (self.args.upload_k + self.args.upload_r)
        model_glob = None

        agr_dict = {}
        vec_list = [0] * (args.upload_k + args.upload_r)
        block_list = [None] * (args.upload_k + args.upload_r)
        for i in range(args.upload_k + args.upload_r):
            block_list[i] = [None] * args.num_users

        flag = (0b1 << args.num_users)-1
        while True:
            shm_name, data_time = self.receive_queue.get()
            data = self.pop_shared_data(shm_name)
            if int(data[3:6]) != self.iter:
                continue
            block_idx = int(data[6:9])
            log.info('Iteration '+ str(self.iter) + " block " + str(block_idx) + " is received at {0} ".format(data_time))  
            if block_idx in part_idx_list:
                continue
            
            if data[0:3] == b'997':
                id = -1
                temp = int.from_bytes(data[9:13], byteorder='big')
                vec_list[block_idx] = vec_list[block_idx] | temp
                log.info('997 block_id:' + str(block_idx) + ' user_idx:' + str(temp))
                while temp:
                    temp = temp >> 1
                    id += 1
                # print('received', block_idx, id)
                block_list[block_idx][id] = pickle.loads(data[13:])

            elif data[0:3] == b'996':
                cur_vec = int.from_bytes(data[9:13], byteorder='big')
                vec_list[block_idx] = vec_list[block_idx] | cur_vec
                log.info('996 block_id:' + str(block_idx) + ' cur_vec:' + str(cur_vec))
            #   print('996 block_id:', block_idx, vec_list[block_idx], 'cur_vec:', cur_vec)
                            
                old_data = agr_dict.get(block_idx)
                if old_data:
                    agr_dict[block_idx] = self.fusion(old_data, data)
                else:
                    agr_dict[block_idx] = data
                # print('is none:', agr_dict is None)
            
            log.info(str(vec_list[block_idx]))
            if vec_list[block_idx] == flag:
                cur_data = agr_dict.get(block_idx)
                if cur_data is not None:
                    temp = int.from_bytes(cur_data[9:13], byteorder='big')
                    cur_data = pickle.loads(cur_data[13:])
                else:
                    temp = 0                
                
                for i in range(args.num_users):
                    if temp & 1 == 0:
                      # print(block_idx, i, type(cur_data), type(block_list[block_idx][i]))
                        cur_data = self.data_fusion(cur_data, block_list[block_idx][i])
                    temp = temp >> 1

                part_idx_list.append(block_idx)
                # print(type(cur_data))
                part_list[block_idx] = cur_data / args.num_users
              # print('aggregated block number:', len(part_idx_list))
                if len(part_idx_list) == self.args.upload_k:
                    pre_code_time = time.time()
                    model_glob = self.decoding(part_list.copy(), part_idx_list.copy(), coding)
                    cur_code_time = time.time()
                    log.info('Iteration '+ str(self.iter) + ". Decoding took about {0} seconds to complete".format(cur_code_time - pre_code_time))
                    return model_glob
                # TODO

    '''
    # upload: 997 + user indx + iteration number + data partition index + data
    def decoder(self, coding, log):
        part_idx_list, part_list = structure(self.args)
        param_list = []
        client_num_wait = 0
        while True:
            data, data_time = self.receive_queue.get()
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
                    param_list.append(torch.tensor(local_model).reshape(-1,1))
                    client_num_wait += 1
            if client_num_wait == self.args.num_users:
                break
        return param_list
    '''

    def update_upload_r(self, t_last, t_cur):
        update_lambda = 1.2
        if t_cur < t_last * update_lambda and self.update_flag == False:
            self.upload_r = max(self.bottom, self.upload_r - self.args.r_delta)
            self.update_p += 1
        elif t_cur > t_last * update_lambda:
            self.upload_r = min(self.args.r_thres, (self.upload_r + self.args.r_thres) / 2)
            self.bottom = min(self.args.num_users + self.bottom, self.args.r_thres)
            self.update_flag = True
            self.update_p = 0
        elif t_cur < t_last / update_lambda and self.update_flag == True:
            self.upload_r = min(self.args.r_thres, (self.upload_r + self.args.r_thres) / 2)
            self.bottom = min(self.args.r_delta + self.bottom, self.args.r_thres)
            self.update_p = 0
        else:
            self.update_flag = False
            self.update_p = 0
        if self.update_p > 5:
            self.bottom = max(self.bottom - self.args.r_delta, 0)
            self.update_p = 0
            
        return self.upload_r

    # download: 999 + iteration number + upload r + data partition index + data
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

                upload_r = str(self.upload_r)  # change iter to str
                upload_r = upload_r.zfill(3)  # change iter into 3 digit

                send_byte = b'999' + bytes(iter_str, encoding="utf8") + bytes(upload_r, encoding="utf8") + model_glob_byte
                key = 'client' + str(j)
                shm_name = self.push_shared_data(send_byte)
                self.send_queue.put((shm_name, key))
                # log.info("Queue Block size is {0} ".format(send_byte.__sizeof__()/(2**20)))
                    
    def processor(self):
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
        t_last = time.time()
        t_cur = 0
        for t in range(self.args.epochs):
            self.iter = t
            params  = get_params_flatten(self.model.state_dict())
            last_time = time.time()
            log.info('Iteration '+ str(t) + ". Distribution starts at {0}".format(last_time))
            self.distribute(params, nc, t, log)
            model_glob = self.decoder(coding, log) #/ self.args.num_users

            for i in range(self.args.num_users):
                self.status_table[i] = 0
            
            cur_time = time.time()
            log.info('Iteration '+ str(t) + ". Uploading ends at {0} ".format(cur_time))
            t_cur = cur_time - last_time
            log.info('Iteration '+ str(t) + ". Operation took around {0} seconds to complete".format(t_cur))
            upload_r = self.update_upload_r(t_last, t_cur)
            log.info('Iteration '+ str(t) + ". New Upload Redundancy r is: {0}".format(upload_r))
            t_last = t_cur
            param_list = [torch.Tensor(model_glob).reshape(-1,1).to(device)]
            model_dict = algorithms.fedavg(param_list, self.model, self.args)
            self.model.load_state_dict(model_dict)
            self.glob_iter.value += 1
            # correct = self.evaluate()
            # log.info(
            #     'Round {:3d}, Test accuracy: {:.2f}'.format(
            #         iter, correct))
            
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
    
    # 997 [0:3] + iteration number [3:6] + data partition index [6:9] + client IO vector [9:13] + data [13:]
    def fusion(self, cur_block, new_block):
        cur_vec = int.from_bytes(cur_block[9:13], byteorder='big')
        new_vec = int.from_bytes(new_block[9:13], byteorder='big')
        gen_vec = cur_vec | new_vec
        
        cur_data = pickle.loads(cur_block[13:])
        new_data = pickle.loads(new_block[13:])
        gen_data = cur_data + new_data

        return b'996' + cur_block[3:9] + gen_vec.to_bytes(length=4, byteorder='big') + pickle.dumps(gen_data)
    
    def data_fusion(self, cur_block, new_block):
        if cur_block is None:
            return new_block
        else:
            gen_block = cur_block + new_block
            return gen_block
        
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
