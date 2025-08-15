import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():          # avoid duplicate handlers in ipython
        return logger
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
