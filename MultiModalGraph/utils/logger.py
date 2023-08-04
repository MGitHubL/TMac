import atexit
import functools
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping

import torch
from termcolor import colored

from MultiModalGraph.utils.file_io import PathManager

__all__ = [
    "setup_logger",
    "log_first_n",
    "_log_api_usage",
    "_ColorfulFormatter",
    "_cached_log_stream",
    "log_every_n_seconds",
    "print_csv_format",
]


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "MultiModalGraph"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg,)

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different MultiModalGraph components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once("MultiModalGraph." + identifier)


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    output=None,
    distributed_rank=0,
    *,
    color=True,
    name="MultiModalGraph",
    abbrev_name=None,
):
    """
    Initialize the MultiModalGraph logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "MultiModalGraph" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "MultiModalGraph" if name == "MultiModalGraph" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = PathManager.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


# _LOG_COUNTER = Counter()
# _LOG_TIMER = {}


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "detectron2"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            logger.info("copypaste: Task: {}".format(task))
            logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
            logger.info(
                "copypaste: "
                + ",".join(["{0:.4f}".format(k[1]) for k in important_res])
            )
        else:
            logger.info(f"copypaste: {task}={res}")
