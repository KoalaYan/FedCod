import grpc
from ..protocol import communication_pb2, communication_pb2_grpc
from concurrent import futures

import multiprocessing as mp
import websockets.client
import websockets.server
import asyncio
import time
import functools
import pickle
from copy import deepcopy
import signal
from threading import Thread
from queue import Queue
import numpy as np
import models

# Modify your Node class to inherit from communication_pb2_grpc.CommunicationServicer
class Node(communication_pb2_grpc.StreamServicer):
    def __init__(self, args, loggername):
        self.args = args
        self.loggername = loggername
        self.server = None

    def create_model(self):
        # define the model architecture
        net_type = self.args.model
        if net_type == 'cnn':
            net = models.CNN().cpu()
        elif net_type == 'resnet20':
            net = models.ResNet(models.ResidualBlock).cpu()
        elif net_type == 'alexnet':
            net = models.AlexNet().cpu()
        elif net_type == 'LR':
            net = models.LR().cpu()
        else:
            raise NotImplementedError
        return net
    

    async def send(self, data, uri):

        def send_stream(data_sent):
            for df in data_sent:
                yield communication_pb2.request(message=df)

        max_size = self.args.block_size
        data_sent = [data[i * max_size : (i + 1) * max_size] for i in range(len(data) // max_size)] + [
            data[max_size * (len(data) // max_size) :]
        ]

        with grpc.insecure_channel(uri) as channel:
            # print(uri)
            # client = communication_pb2_grpc.StreamStub(channel)  # 客户端使用Stub类发送请求,参数为频道,为了绑定链接
            # response = client.msgStream(self.send_stream(data_sent))  # 需要将上面的send_stream传进来
            stub = communication_pb2_grpc.StreamStub(channel)
            response = stub.MsgStream(send_stream(data_sent))
            # print(f"Server response: {response.message}")

    async def lc_send(self, data, channel):
        def send_stream(data_sent):
            for df in data_sent:
                yield communication_pb2.request(message=df)

        max_size = self.args.block_size
        data_sent = [data[i * max_size : (i + 1) * max_size] for i in range(len(data) // max_size)] + [
            data[max_size * (len(data) // max_size) :]
        ]

        stub = communication_pb2_grpc.StreamStub(channel)
        response = stub.MsgStream(send_stream(data_sent))


    def MsgStream(self, request_iterator, context):
        data_chunks = []

        for data_chunk in request_iterator:
            data_chunks.append(data_chunk.message)

        received_data = b''.join(data_chunks)
        
        # print(f"Received {len(received_data)} bytes of data.")


        cur_time = time.time()
        self.receive_queue.put((received_data, cur_time))
        return communication_pb2.response(message=b'ok')

    async def manager(self, log):
        self.send_queue = mp.Queue()
        self.receive_queue = mp.Queue()
        process_list = []
        
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

