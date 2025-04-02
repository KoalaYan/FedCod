import multiprocessing as mp
import asyncio

from utils.config_to_arg import argument
from utils.logger import create_logger

from device.client.base_client import Client
from device.client.coding_client import CodClient
from device.client.nc_client import NCClient
from device.client.nc_d_client import NCDClient
from device.client.nc_preagr_nowait_client import NCAGRClient
from device.client.nc_preagr_wait_client import NCAGRWClient
from device.client.nc_d_preagr_wait_client import NCDAGRWClient
from device.client.nc_preagr_dynamic_client import NCAGRClient as DNCAGRClient
from device.client.nc_d_preagr_dynamic_client import NCDAGRClient as DNCDAGRClient

from device.client.nc_down_client import NCDownClient
from device.client.nc_d_down_client import NCDDownClient
from device.client.nc_upload_client import NCUploadClient
from device.client.preagr_wait_client import AGRWClient
from device.client.preagr_nowait_client import AGRClient
from device.client.hier_client import HierClient
from device.client.hier_center_client import HierCenterClient
from device.client.all_reduce_client import AR_Client

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = argument()
    loggername = './log/' + args.commu + '/' + args.config_section + '.log'
    if args.commu == 'hier':        
        if args.config_section == 'hierserver':
            loggername = './log/' + args.commu + '/' + args.config_section + args.cfg_fn[-1] + '.log'
        else:
            loggername = './log/' + args.commu + '/client_' + args.cfg_fn[-1] + '_' + str(args.idx_users) + '.log'

    log = create_logger(loggername)
    log.info(args.config_section +' '+ args.client_ip +' '+ args.client_port)
    if args.commu == 'default':
        client = Client(args, loggername)
    # elif args.commu == 'lc_default':
    #     client = LgCnctClient(args, loggername)
    elif args.commu == 'cod':
        # client = CodClient(args, loggername)
        client = CodClient(args, loggername)
    elif args.commu == 'nc':
        client = NCClient(args, loggername)
    elif args.commu == 'ncd':
        client = NCDClient(args, loggername)
    elif args.commu == 'ncagr':
        client = NCAGRClient(args, loggername)
    elif args.commu == 'ncagrw':
        client = NCAGRWClient(args, loggername)
    elif args.commu == 'ncdagrw':
        client = NCDAGRWClient(args, loggername)
    elif args.commu == 'dncagr':
        client = DNCAGRClient(args, loggername)
        
    elif args.commu == 'ncddown':
        client = NCDDownClient(args, loggername)
    elif args.commu == 'ncdown':
        client = NCDownClient(args, loggername)
    elif args.commu == 'ncupload':
        client = NCUploadClient(args, loggername)
    elif args.commu == 'agrw':
        client = AGRWClient(args, loggername)
    elif args.commu == 'agr':
        client = AGRClient(args, loggername)
    elif args.commu == 'dncdagr':
        client = DNCDAGRClient(args, loggername)

    elif args.commu == 'hier':        
        if args.config_section == 'hierserver':
            client = HierCenterClient(args, loggername)
        else:
            client = HierClient(args, loggername)
            
    elif args.commu == "allreduce":
        client = AR_Client(args, loggername)
    # elif args.commu == 'fedcod':
    #     client = FedCodClient(args, loggername)

    log.info("Initializing..." + args.config_section)
    asyncio.run(client.manager(log))