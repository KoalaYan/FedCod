import multiprocessing as mp
from multiprocessing import shared_memory
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

class Server(Node, communication_pb2_grpc.StreamServicer):
    def __init__(self, args, loggername):
        super().__init__(args, loggername)
        self.address_dict = {}
        for i in range(self.args.num_users):
            client_name = 'client' + str(i)
            self.address_dict[client_name] = [args.client_ip_list[i], args.client_port_list[i]]


    def load_dataset(self):
        #  load the dataset
        if self.args.dataset == 'FashionMNIST':
            test_data = DataLoader(torchvision.datasets.FashionMNIST(root = './data/', train=False, download=True, transform=transforms.ToTensor()), 250, drop_last = True, shuffle=False)
        elif self.args.dataset == 'CIFAR10':        
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            trans1=transforms.Compose([
                transforms.ToTensor(),
                normalize])
            trans2=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            test_data = DataLoader(torchvision.datasets.CIFAR10(root = './data/', train=False, download=True, transform=trans2), 250, drop_last = True, shuffle=False)
        elif self.args.dataset == 'HAR':
            _, X_test, _, y_test = HAR_dataloader()
            X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)
            test_data = DataLoader(TensorDataset(X_test, y_test), 368, drop_last = True, shuffle=False)
        else:
            raise NotImplementedError
        return test_data
    
    def evaluate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = self.model.to(device)
        correct = 0
        with torch.no_grad():
            net.eval()
            for data, target in self.dataset:
                data, target = data.to(device), target.to(device)
                output = net(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        correct = 1. * correct / len(self.dataset.dataset)
        net.train()
        # self.model.cpu()
        return correct
    
    
    def sender(self):
        asyncio.run(self.parallel_senders())

    def processor(self):
        asyncio.run(self.async_processor())

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
            shm_name = self.send_queue.get()
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
                
                
            for key in self.address_dict.keys():
                shm_name = self.push_shared_data(data)
                queue_list[key].put(shm_name)

        for t in thread_list:
            t.join()
        for key in queue_list.keys():
            queue_list[key].join()

    async def async_processor(self):
        log = create_logger(self.loggername)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # pretrain_model_dict = torch.load('./resnet152_cifar10_2.pth')
        self.model = self.create_model().to(device)
        self.model.train()
        # self.model.load_state_dict(pretrain_model_dict)
        self.dataset = self.load_dataset()
        last_time = time.time()
        for iter in range(self.args.epochs):
            params = np.array(get_params_flatten(self.model.state_dict()), dtype=np.float64)
            cur_time = time.time()
            # print('-------------Iteration',iter,'-------------')
            log.info('Iteration '+ str(iter) + ". Operation took around {0} seconds to complete".format(cur_time - last_time))
            last_time = cur_time
            cur_dis_time = time.time()
            log.info('Iteration '+ str(iter) + ". Distribution starts at {0} ".format(cur_dis_time))
            
            shm_data = pickle.dumps(params)
            shm_name = self.push_shared_data(shm_data)
            self.send_queue.put(shm_name)

            count = 0
            param_list = []
            while count < self.args.num_users:
                shm_name, rec_time = self.receive_queue.get()
                data = self.pop_shared_data(shm_name)
                client_idx = int(data[:3])
                log.info('Iteration '+ str(iter) + ". Client " + str(client_idx) + " is received at {0} ".format(rec_time))
                count += 1
                param_list.append(pickle.loads(data[3:]))
                
            cur_up_time = time.time()
            log.info('Iteration '+ str(iter) + ". Uploading ends at {0} ".format(cur_up_time))
            param_list = [torch.Tensor(x).reshape(-1,1).to(device) for x in param_list]
            cur_2_time = time.time()
            model_dict = algorithms.fedavg(param_list, self.model, self.args)
            cur_1_time = time.time()
            log.info('Iteration '+ str(iter) + ". Aggregation took around {0} seconds to complete".format(cur_1_time - cur_2_time))
            self.model.load_state_dict(model_dict)
            correct = self.evaluate()
            log.info(
                'Round {:3d}, Test accuracy: {:.2f}'.format(
                    iter, correct))
            
        shm_name = self.push_shared_data(b'000')
        self.send_queue.put(shm_name)
        time.sleep(5)
        os.kill(os.getppid(), signal.SIGTERM)
