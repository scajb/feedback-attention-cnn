import logging
import os
import time
import traceback
from pathlib import Path


def log_info(*argv: object) -> object:
    log_info_list(argv)


def log_error(msg):
    print_time_and_message(msg)
    logging.error(msg)


def log_exception(ex, extra_msg=""):
    stack_trace = "".join(traceback.TracebackException.from_exception(ex).format())
    msg = f"{extra_msg}:{os.linesep}{stack_trace}" if extra_msg else stack_trace
    log_error(msg)


def print_time_and_message(msg):
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(current_time, msg)


def log_info_list(mylist):
    msg = " ".join(map(str, mylist))
    print_time_and_message(msg)
    logging.info(msg)


def log_version():
    with open(f"{Path(__file__).parent}/_version.txt") as vf:
        log_info(f"Code version: {vf.readline().rstrip()}")


def init_logging(log_path, argv=None):
    """
    Configures Python logging package to write to given log file.

    :param log_path: Ouput log file path
    :param argv: Optional script parameters to log at top of file
    :return: None
    """

    dir_path = os.path.dirname(log_path)
    dir_created = False
    if not os.path.exists(dir_path):
        os.umask(0)
        os.makedirs(dir_path)
        dir_created = True

    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    if dir_created:
        log_info(f"Created directory for log file output at {dir_path}")

    log_info(f"STARTED LOGGING to {log_path}")
    log_version()

    if argv:
        log_info(*argv)
