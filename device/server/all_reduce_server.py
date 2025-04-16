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
from scipy.optimize import linprog

from concurrent import futures
import grpc
from ..protocol import communication_pb2, communication_pb2_grpc
from .base_server import Server


class AR_Server(Server):
    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.iter = 0

    def solve_bandwidth_allocation(self, bandwidth_matrix):
        P, N = bandwidth_matrix.shape

        # 计算每列和每行的最小值
        min_bij_col = np.min(bandwidth_matrix, axis=0)  # 每列的最小值 (scatter)
        min_bij_row = np.min(bandwidth_matrix, axis=1)  # 每行的最小值 (multicast)

        # 定义线性规划问题
        c = [1] + [0] * N  # 目标函数：最小化 T
        A = []
        b = []

        # 添加 T >= x_j / min_bij_col[j] 的约束
        for j in range(N):
            constraint = [0] * (N + 1)
            constraint[0] = -1  # -T
            constraint[j + 1] = 1 / min_bij_col[j]  # + x_j / min_bij_col[j]
            A.append(constraint)
            b.append(0)

        # 添加 T >= x_j / min_bij_row[j] 的约束
        for j in range(N):
            constraint = [0] * (N + 1)
            constraint[0] = -1  # -T
            constraint[j + 1] = 1 / min_bij_row[j]  # + x_j / min_bij_row[j]
            A.append(constraint)
            b.append(0)

        # 添加 sum(x) = 1 的约束
        A_eq = [[0] + [1] * N]
        b_eq = [1]

        # 添加 x_i >= 0 的约束
        bounds = [(0, None)] * (N + 1)

        # 求解线性规划
        result = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            return result.x[1:]  # 返回 x 的值（去掉 T）
        else:
            raise ValueError("线性规划求解失败")
    
    # ack and sync profiling result: 666 + iteration number + client index + client IO vector + data
    def bandwidth_sync(self, iter, log):
        args = self.args

        bandwidths_list = [None] * args.num_users
        flag = np.zeros((args.num_users,), dtype=bool)

        while True:
            shm_name, data_time = self.receive_queue.get()
            data = self.pop_shared_data(shm_name)
            # print(int(data[3:6]), iter)
            if int(data[3:6]) != iter:
                continue
            client_idx = int(data[6:9])
            log.info('Iteration '+ str(iter) + " ack of " + str(client_idx) + " is received at {0} ".format(data_time))  
            
            bandwidths_list[client_idx] = pickle.loads(data[9:])
            flag[client_idx] = True
            if np.sum(flag) == args.num_users:
                break
        
        commu_x_list = self.solve_bandwidth_allocation(np.array(bandwidths_list))
        # print(commu_x_list)
        print("results:", commu_x_list)
                
        commu_x_list_byte = pickle.dumps(commu_x_list)

        iter_str = str(iter)  # change iter to str
        iter_str = iter_str.zfill(3)  # change iter into 3 digit
        send_byte = b'111' + bytes(iter_str, encoding="utf8") + commu_x_list_byte
        for j in range(self.args.num_users):
            key = 'client' + str(j)
            shm_name = self.push_shared_data(send_byte)
            self.send_queue.put((shm_name, key))

    # download: 999 + iteration number + data partition index + data
    def distribute(self, iter, log):
        params = np.array(get_params_flatten(self.model.state_dict()), dtype=np.float64)
        cur_dis_time = time.time()
        log.info('Iteration '+ str(iter) + ". Distribution starts at {0} ".format(cur_dis_time))
        
        shm_data = pickle.dumps(params)
        iter_str = str(iter)  # change iter to str
        iter_str = iter_str.zfill(3)  # change iter into 3 digit
        msg_byte = b'222' + bytes(iter_str, encoding="utf8") + shm_data
        shm_name = self.push_shared_data(msg_byte)
        self.send_queue.put((shm_name, 'all'))
                    
    async def async_processor(self):
        log = create_logger(self.loggername)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pretrain_model_dict = torch.load('./resnet152_cifar10_2.pth')
        self.model = self.create_model().to(device)
        self.model.train()
        # self.model.load_state_dict(pretrain_model_dict)
        self.dataset = self.load_dataset()
        last_time = time.time()
        self.distribute(0, log)
        # print("Data distribution done.")
        # self.bandwidth_sync(0, log)
        # print("Bandwidth sync done.")
        for t in range(self.args.epochs):
            self.bandwidth_sync(t, log)
            print("Iteration ", t, ": Bandwidth sync done.")
            
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
            # if int(data[3:6]) != self.glob_iter.value:
            #     print("Data is not for this iteration, ignore it.", self.glob_iter.value, int(data[3:6]))
            #     continue
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
            cur_time = time.time()
            log.info("Gotten data from send queue at {0} ".format(cur_time))
            if data == b'000':
                # server shutdown
                for key in self.address_dict.keys():
                    shm_name = self.push_shared_data(data)
                    queue_list[key].put(shm_name)
                    websocket_num -= 1
                    if websocket_num == 0:
                        break
                    continue
            elif data[:3] == b'111':
                shm_name = self.push_shared_data(data)
                queue_list[target].put(shm_name)
                continue
            elif data[:3] == b'222':
                for key in self.address_dict.keys():
                    shm_name = self.push_shared_data(data)
                    queue_list[key].put(shm_name)
                continue

            # if int(data[3:6]) != self.glob_iter.value:
            #     continue

            # shm_name = self.push_shared_data(data)
            # queue_list[target].put(shm_name)

    def MsgStream(self, request_iterator, context):
        data_chunks = []

        for data_chunk in request_iterator:
            data_chunks.append(data_chunk.message)

        data = b''.join(data_chunks)
        cur_time = time.time()

        # print(f"Received {len(data)} bytes of data.")
        shm_name = self.push_shared_data(data)
        self.receive_queue.put((shm_name,cur_time))
        # if int(data[3:6]) == self.glob_iter.value:
        #     shm_name = self.push_shared_data(data)
        #     self.receive_queue.put((shm_name,cur_time))
        # else:
        #     print("Data is not for this iteration, ignore it.", self.glob_iter.value, int(data[3:6]))
            
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
