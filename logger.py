"""
Logger taken from Ze Liu's Swin Transformer https://github.com/microsoft/Swin-Transformer
"""
import os
import sys
import logging
import functools
from termcolor import colored
@functools.lru_cache()
def create_logger(output_dir, dataset, save_to_file, dist_rank=0, model_type="", model_name=""):
    # create logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    if save_to_file:
        # create file handlers
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f'log_{dataset}_{model_type}_{model_name}.txt'), mode='w'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger