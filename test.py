import time
import numpy as np
import configparser
import torch
import torch.nn as nn
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--commu', type=str, default='default', help='communication arch: default/fedcod')
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--q', type=float, default=0.5, help="q value")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha value of the dirichlet distribution")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--idx_users', type=int, default=0, help="the index of each client")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # coding arguments
    parser.add_argument('--upload_k', type=int, default=6, help='the value of k in uploading')
    parser.add_argument('--upload_r', type=int, default=4, help='the number of stragglers in uploading')
    parser.add_argument('--download_k', type=int, default=6, help='the value of k in downloading')
    parser.add_argument('--download_r', type=int, default=4, help='the number of stragglers in downloading')

    # other arguments
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help="name of dataset")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_dirichlet', action='store_true', help='whether use dirichlet or not')
    parser.add_argument('--noniid_q', action='store_true', help='whether use dirichlet or not')
    parser.add_argument('--noniid_quantity', action='store_true', help='whether use dirichlet or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--num_process', type=int, default=10, help='number of client processes')
    parser.add_argument('--num_users_node', type=int, default=100, help="number of users of each physical node")
    parser.add_argument('--wait_time', type=int, default=10, help="the waiting of each client before sending a request")
    parser.add_argument('--window_size', type=int, default=100, help="the size of each partition window")
    parser.add_argument('--selection_alpha', type=float, default=0.2, help="the fraction to be replaced in forwarder selection")

    # config
    parser.add_argument('--client_ip', type=str, default='127.0.0.2', help="client ip address")
    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help="server ip address")
    parser.add_argument('--client_port', type=str, default='8000', help="client tcp port")
    parser.add_argument('--server_port', type=str, default='9999', help="server ip address")
    parser.add_argument('--client_ip_list', nargs='+', default=['127.0.0.2'])
    parser.add_argument('--client_port_list', nargs='+', default=['8000'])
    parser.add_argument('--config_section', type=str, default='client0', help="the section of the cfg.ini")
    parser.add_argument('--block_size', type=int, default=2**30, help='the size of data frame/block')
    args = parser.parse_args()
    return args

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

def argument():
    conf = configparser.ConfigParser()
    conf.read('cfg.ini')
    args = args_parser()
    args.dataset = conf.get('parameter', 'dataset')
    args.model = conf.get('parameter', 'model')
    args.num_channels = conf.getint('parameter', 'num_channels')
    args.epochs = conf.getint('parameter', 'epochs')
    args.frac = conf.getint('parameter', 'frac')
    args.gpu = conf.getint('parameter', 'gpu')
    args.num_users = conf.getint('parameter', 'num_users')
    args.upload_k = conf.getint('parameter', 'upload_k')
    args.upload_r = conf.getint('parameter', 'upload_r')
    args.download_k = conf.getint('parameter', 'download_k')
    args.download_r = conf.getint('parameter', 'download_r')
    args.window_size = conf.getint('parameter', 'window_size')
    args.block_size = int(conf.getfloat('parameter', 'block_size'))
    args.server_ip = conf.get('server', 'server_ip')
    args.iid = conf.getboolean('parameter', 'iid')
    if args.config_section != 'server':
        args.idx_users = conf.getint(args.config_section, 'idx')
        args.client_port = conf.get(args.config_section, 'client_port')
        args.client_ip = conf.get(args.config_section, 'client_ip')
        args.my_ip = args.client_ip
        args.my_port = args.client_port
    else:
        args.my_ip = args.server_ip
        args.my_port = args.server_port
        args.client_ip_list = []
        args.client_port_list = []
        for idx in range(args.num_users):
            config_section = 'client'+str(idx)
            args.client_ip_list.append(conf.get(config_section, 'client_ip'))
            args.client_port_list.append(conf.get(config_section, 'client_port'))
    return args


def get_params_flatten(model_dict):
    param_1 = np.array([])
    for key in model_dict.keys():
        if "num_batches_tracked" not in key:
            item = model_dict[key].numpy().ravel()
            param_1 = np.append(param_1, item)
    return param_1

class Coding:
    def Cauchy(self, m, n):
        x = np.array(range(n + 1, n + m + 1))
        y = np.array(range(1, n + 1))
        x = x.reshape((-1, 1))
        diff_matrix = x - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = np.identity(k)
        P = self.Cauchy(n - k, k)
        return np.concatenate((I, P), axis=0)

    def multiply(self, M, G):
        count = 0
        D = M[0].shape
        X = 1
        Y = D[-1]
        N, K = G.shape
        R = np.zeros((N, X, Y))
        for i in range(N):
            for j in range(K):
                if G[i, j] != 0:
                    R[i] = R[i] + G[i, j] * M[j]
                    count += 1
                    
        return R

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        if M.shape[0] % k != 0:
            zeros = np.zeros(k-M.shape[0] % k)
            M = np.append(M, zeros)
        M = M.reshape(-1)
        M = np.array_split(M, k)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        return self.multiply(M, np.linalg.inv(G))


class OptimizedCoding:
    def Cauchy(self, m, n):
        x = np.arange(n + 1, n + m + 1)
        y = np.arange(1, n + 1)
        diff_matrix = x[:, np.newaxis] - y
        cauchym = 1.0 / diff_matrix
        return cauchym

    def RS(self, n, k):
        I = np.identity(k)
        P = self.Cauchy(n - k, k)
        return np.concatenate((I, P), axis=0)

    def multiply(self, M, G):
        R = np.matmul(G, M)
        return R[:, np.newaxis, :]

    def encode_RS(self, M, k, r):
        G = self.RS(k + r, k)
        if M.shape[0] % k != 0:
            zeros = np.zeros(k - M.shape[0] % k)
            M = np.append(M, zeros)
        M = M.reshape(-1)
        M = np.array_split(M, k)
        return self.multiply(M, G)

    def decode_RS(self, M, k, r, index):
        G = self.RS(k + r, k)
        G = G[index]
        I = np.identity(k)
        inv = np.linalg.solve(G, I)
        return self.multiply(M, inv)


def test1_enc(args, coding, arr_glob, times):
    time_list = []
    for i in range(times):
        pre_code_time = time.time()
        model_glob = coding.encode_RS(arr_glob, args.upload_k, args.upload_r)
        cur_code_time = time.time()
        time_list.append((cur_code_time - pre_code_time))
    
    t = np.average(time_list)
    print("Original encoding took about {0} seconds to complete (Averaged Result)".format(t))
    return model_glob
        
def test2_enc(args, coding, arr_glob, times):
    time_list = []
    for i in range(times):
        pre_code_time = time.time()
        model_glob = coding.encode_RS(arr_glob, args.upload_k, args.upload_r)
        cur_code_time = time.time()
        time_list.append((cur_code_time - pre_code_time))
    t = np.average(time_list)
    print("Optimized encoding took about {0} seconds to complete (Averaged Result)".format(t))
    return model_glob
 
def test1_dec(args, coding, model_glob_list, order_list, times):
    time_list = []
    for i in range(times):
        pre_code_time = time.time()
        model_glob = coding.decode_RS(model_glob_list, args.download_k, args.download_r, order_list)
        cur_code_time = time.time()
        time_list.append((cur_code_time - pre_code_time))
    
    t = np.average(time_list)
    print("Original decoding took about {0} seconds to complete (Averaged Result)".format(t))
    return model_glob
        
def test2_dec(args, coding, model_glob_list, order_list, times):
    time_list = []
    for i in range(times):
        pre_code_time = time.time()
        model_glob = coding.decode_RS(model_glob_list, args.download_k, args.download_r, order_list)
        cur_code_time = time.time()
        time_list.append((cur_code_time - pre_code_time))
    t = np.average(time_list)
    print("Optimized decoding took about {0} seconds to complete (Averaged Result)".format(t))
    return model_glob

if __name__ == '__main__':
    args = argument()
    model = AlexNet().cpu()
    params  = get_params_flatten(model.state_dict())
    arr_glob = params.reshape(-1)
    repeat_times = 5

    coding = OptimizedCoding()
    R2 = test2_enc(args, coding, arr_glob, repeat_times)

    print("------------------")
    coding = Coding()
    R1 = test1_enc(args, coding, arr_glob, repeat_times)

    print("Encoding Result is: ", np.sum(R1-R2)<1e-10)


    part_list = R1
    order_list = np.random.choice(args.upload_k+args.upload_r, args.upload_k, replace=False)
    print(order_list)
    model_glob = np.array([])
    for i in order_list:
        model_glob = np.append(model_glob, part_list[i])
    model_glob = model_glob.reshape(args.download_k, -1)


    coding = OptimizedCoding()
    R2 = test2_dec(args, coding, model_glob, order_list, repeat_times)
    model_glob_2 = R2.reshape(-1)

    print("------------------")
    coding = Coding()
    R1 = test1_dec(args, coding, model_glob, order_list, repeat_times)
    model_glob_1 = R1.reshape(-1)

    print("Decoding Result is: ", np.sum(model_glob_1-model_glob_2)<1e-10)
