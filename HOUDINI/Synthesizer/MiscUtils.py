import inspect
import logging
import logging.config
import os

import time

import datetime
from functools import wraps

import yaml


def getUniqueFn():
    count = 0

    def inner():
        nonlocal count
        count += 1
        return count

    return inner


dbgn = getUniqueFn()


def nextId(id: int) -> int:
    return id + 1


dbgPrintCounter = getUniqueFn()


def dbgPrint(msg):
    cf = inspect.currentframe()
    bcf = cf.f_back
    lineNo = bcf.f_lineno
    bcfi = inspect.getframeinfo(bcf)
    filename = bcfi.filename
    sn = dbgPrintCounter()
    print('File "%s", line %d, sn %d \n %s' % (filename, lineNo, sn, msg))
    return sn


g_start_time = time.time()


def getElapsedTime() -> str:
    etime_sec = time.time() - g_start_time
    return formatTime(etime_sec)


def formatTime(time_sec) -> str:
    et = datetime.timedelta(seconds=time_sec)
    return str(et)


def setup_logging(
        default_path='logging.yaml',
        default_level=logging.DEBUG,
        env_key='LOG_CFG'
):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def logEntryExit(name):
    def logEntryExitName(method):
        @wraps(method)
        def impl(*args, **kwargs):
            print('START %s:' % name)
            res = method(*args, **kwargs)
            print('END %s:' % name)
            # print('progs: ', progUnks)
            return res

        return impl
    return logEntryExitName
