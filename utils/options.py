import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_fn', type=str, default='cfg', help='configure file name')
    # federated arguments
    parser.add_argument('--commu', type=str, default='default', help='communication arch: default/fedcod')
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--q', type=float, default=0.5, help="q value")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha value of the dirichlet distribution")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
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
    parser.add_argument('--hierserver_ip', type=str, default='127.0.0.3', help="hierarchical server ip address")
    parser.add_argument('--client_port', type=str, default='8000', help="client tcp port")
    parser.add_argument('--server_port', type=str, default='9999', help="server ip address")
    parser.add_argument('--hierserver_port', type=str, default='9999', help="hierarchical server ip address")
    parser.add_argument('--client_ip_list', nargs='+', default=['127.0.0.2'])
    parser.add_argument('--client_port_list', nargs='+', default=['8000'])
    parser.add_argument('--config_section', type=str, default='client0', help="the section of the cfg.ini")
    parser.add_argument('--block_size', type=int, default=2**30, help='the size of data frame/block')
    args = parser.parse_args()
    return args