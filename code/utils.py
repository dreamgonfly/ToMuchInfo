import logging
from datetime import datetime
from os.path import dirname, abspath, join, exists
import os


def get_logger(model_name):
    base_dir = dirname(abspath(__file__))
    log_dir = join(base_dir, 'logs')
    #    LOG_DIR = '/output'
    if not exists(log_dir):
        os.mkdir(log_dir)

    log_filename = '{model_name}-{datetime}.log'.format(model_name=model_name, datetime=datetime.now())
    log_filepath = join(log_dir, log_filename)
    logger = logging.getLogger('logger')
    if not logger.handlers:  # execute only if logger doesn't already exist
        fileHandler = logging.FileHandler(log_filepath.format(datetime=datetime.now()))
        streamHandler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%m-%d %H:%M:%S')

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
        logger.setLevel(logging.INFO)

    return logger