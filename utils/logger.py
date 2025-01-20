import multiprocessing as mp
import logging
from logging import handlers

# class Logger(object):
#     level_relations = {
#         'debug':logging.DEBUG,
#         'info':logging.INFO,
#         'warning':logging.WARNING,
#         'error':logging.ERROR,
#         'crit':logging.CRITICAL
#     }

#     def __init__(self,loggername, filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', mode='a'):
#         self.logger = logging.getLogger(loggername)
#         self.logger = mp.get_logger(loggername)
#         format_str = logging.Formatter(fmt)#format
#         self.logger.setLevel(self.level_relations.get(level))#level
#         sh = logging.StreamHandler()#output to terminal
#         sh.setFormatter(format_str) #terminal format
#         # th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#logging file
#         th = logging.FileHandler(filename=filename, mode=mode, encoding='utf-8')#logging file
#         # interval，backupCount = number of files，when = S, M, H, D, W, midnight
    
#         th.setFormatter(format_str)#file format
#         self.logger.addHandler(sh)
#         self.logger.addHandler(th)

def create_logger(loggername):
    # import multiprocessing, logging
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(\
        '[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    handler = logging.FileHandler(loggername)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers): 
        logger.addHandler(handler)
    return logger