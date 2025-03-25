import configparser
from utils.options import args_parser


def argument():
    conf = configparser.ConfigParser()
    args = args_parser()
    conf.read('configs/'+args.cfg_fn+'.ini')
    args.dataset = conf.get('parameter', 'dataset')
    args.model = conf.get('parameter', 'model')
    args.num_channels = conf.getint('parameter', 'num_channels')
    args.epochs = conf.getint('parameter', 'epochs')
    args.frac = conf.getint('parameter', 'frac')
    args.gpu = conf.getint('parameter', 'gpu')
    args.num_users = conf.getint('parameter', 'num_users')
    args.upload_k = conf.getint('parameter', 'upload_k')
    args.upload_r = conf.getint('parameter', 'upload_r')
    args.r_thres = conf.getint('parameter', 'r_thres')
    args.r_delta = conf.getint('parameter', 'r_delta')
    args.download_k = conf.getint('parameter', 'download_k')
    args.download_r = conf.getint('parameter', 'download_r')
    args.window_size = conf.getint('parameter', 'window_size')
    args.block_size = int(conf.getfloat('parameter', 'block_size'))
    args.server_ip = conf.get('server', 'server_ip')
    args.server_port = conf.get('server', 'server_port')
    args.iid = conf.getboolean('parameter', 'iid')

    if args.commu == 'hier' and args.config_section != 'server':
        args.hierserver_ip = conf.get('hierserver', 'hierserver_ip')
        args.hierserver_port = conf.get('hierserver', 'hierserver_port')

    if args.config_section == 'hierserver':
        args.my_ip = conf.get('hierserver', 'local_ip')
        args.my_port = conf.get('hierserver', 'hierserver_port')
        args.client_ip_list = []
        args.client_port_list = []
        for idx in range(args.num_users):
            config_section = 'client'+str(idx)
            args.client_ip_list.append(conf.get(config_section, 'client_ip'))
            args.client_port_list.append(conf.get(config_section, 'client_port'))
    
    elif args.config_section != 'server':
        args.idx_users = conf.getint(args.config_section, 'idx')
        args.client_port = conf.get(args.config_section, 'client_port')
        args.client_ip = conf.get(args.config_section, 'client_ip')
        args.my_ip = conf.get(args.config_section, 'local_ip')
        args.my_port = args.client_port
    else:
        args.my_ip = conf.get('server', 'local_ip')
        args.my_port = args.server_port
        args.client_ip_list = []
        args.client_port_list = []
        for idx in range(args.num_users):
            config_section = 'client'+str(idx)
            args.client_ip_list.append(conf.get(config_section, 'client_ip'))
            args.client_port_list.append(conf.get(config_section, 'client_port'))
    return args
