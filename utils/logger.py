# coding: utf-8

import logging
import os
from uuid import uuid4
from utils.utils import get_local_time


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    LOGROOT = config['log_dir'] or 'log'
    dir_name = LOGROOT
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Independent runs/frozen evaluations may start within the same second.
    # Never truncate another experiment's log on a filename collision.
    seed = config['seed']
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
    logfilename = '{}-{}-seed{}-{}-{}.log'.format(
        config['model'], config['dataset'], seed, get_local_time(), uuid4().hex[:8])

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    # comment following 3 lines and handlers = [sh, fh] to cancel file dump.
    fh = logging.FileHandler(logfilepath, 'x', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        #handlers=[sh]
        force=True,
        handlers = [sh, fh]
    )



    return os.path.abspath(logfilepath)
