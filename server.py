
import multiprocessing as mp
import asyncio

from utils.config_to_arg import argument
from utils.logger import create_logger

from device.server.base_server import Server
from device.server.coding_server import CodServer
from device.server.nc_server import NCServer
from device.server.nc_d_server import NCDServer
from device.server.nc_preagr_nowait_server import NCAGRServer
from device.server.nc_preagr_wait_server import NCAGRWServer
from device.server.nc_d_preagr_wait_server import NCDAGRWServer
from device.server.nc_preagr_dynamic_server import NCAGRServer as DNCAGRServer
from device.server.nc_d_preagr_dynamic_server import NCDAGRServer as DNCDAGRServer

from device.server.nc_down_server import NCDownServer
from device.server.nc_d_down_server import NCDDownServer
from device.server.nc_upload_server import NCUploadServer
from device.server.preagr_wait_server import AGRWServer
from device.server.preagr_nowait_server import AGRServer
from device.server.hier_server import HierServer

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = argument()
    loggername = './log/' + args.commu + '/' + args.config_section + '.log'
    log = create_logger(loggername)
    log.info(args.config_section +' '+ args.server_ip +' '+ args.server_port)
    if args.commu == 'default':
        server = Server(args, loggername)
    # elif args.commu == 'lc_default':
    #     server = LgCnctServer(args, loggername)
    elif args.commu == 'cod':
        server = CodServer(args, loggername) 
    elif args.commu == 'nc':
        server = NCServer(args, loggername)
    elif args.commu == 'ncd':
        server = NCDServer(args, loggername)
    elif args.commu == 'ncagr':
        server = NCAGRServer(args, loggername)
    elif args.commu == 'ncagrw':
        server = NCAGRWServer(args, loggername)
    elif args.commu == 'ncdagrw':
        server = NCDAGRWServer(args, loggername)
    elif args.commu == 'dncagr':
        server = DNCAGRServer(args, loggername)


    elif args.commu == 'ncddown':
        server = NCDDownServer(args, loggername)
    elif args.commu == 'ncdown':
        server = NCDownServer(args, loggername)
    elif args.commu == 'ncupload':
        server = NCUploadServer(args, loggername)
    elif args.commu == 'agrw':
        server = AGRWServer(args, loggername)
    elif args.commu == 'agr':
        server = AGRServer(args, loggername)
    elif args.commu == 'dncdagr':
        server = DNCDAGRServer(args, loggername)
    
    elif args.commu == 'hier' and args.config_section == 'server':
        loggername = './log/' + args.commu + '/' + args.config_section + '.log'
        server = HierServer(args, loggername)
    
    # elif args.commu == 'fedcod':
    #     server = FedCodServer(args, loggername) 
    log.info("Initializing..." + args.config_section)
    asyncio.run(server.manager(log))