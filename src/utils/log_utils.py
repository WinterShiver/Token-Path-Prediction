import logging
import os
from pathlib import Path
import datetime


def create_logger(log_file_name='model.log'):
    # 定义本地日志
    fhandler = logging.FileHandler(filename=log_file_name, mode='a')
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    fhandler.setFormatter(formatter)
    local_logger = logging.getLogger()
    local_logger.addHandler(fhandler)
    local_logger.setLevel(logging.INFO)
    return local_logger


def create_logger_v2(log_dir='./'):
    # 目录不存在则创建
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # 使用当前时间命名
    log_file_name = 'model_log.txt'
    log_file_path = os.path.join(log_dir, log_file_name)
    # 定义本地日志
    fhandler = logging.FileHandler(filename=log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
    fhandler.setFormatter(formatter)
    local_logger = logging.getLogger()
    local_logger.addHandler(fhandler)
    local_logger.setLevel(logging.INFO)
    return local_logger
