"""This module contains the logging setup for the application."""

import logging
import os
import sys
import time
from datetime import datetime
from io import TextIOWrapper

from telcorain.handlers import config_handler

logger = logging.getLogger("telcorain")


class InitLogHandler(logging.Handler):
    """
    Custom logging handler for buffering log messages during application initialization,
    while also printing them to stdout or stderr.
    """

    def __init__(self):
        """Initialize the handler with an empty buffer."""
        super().__init__()
        self.buffer = []

    def emit(self, record):
        """
        Emit a log message and print it to stdout or stderr.
        :param record: LogRecord object
        """
        self.buffer.append(record)
        msg = self.format(record)
        # print DEBUG, INFO, WARNING to stdout, while ERROR and CRITICAL to stderr
        if record.levelno < logging.ERROR:
            print(msg, file=sys.stdout)
        else:
            print(msg, file=sys.stderr)


def setup_init_logging(logger, logs_dir: str = "./logs") -> None:
    """
    Set up the initialization logging handler for the application.
    :return: InitLogHandler object
    """
    init_logger = InitLogHandler()
    init_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    init_formatter.converter = time.gmtime  # use UTC time
    init_logger.setFormatter(init_formatter)
    logger.addHandler(init_logger)
    logger.setLevel(config_handler.read_option("logging", "init_level"))
    sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    setup_file_logging(logger, logs_dir)


def setup_file_logging(logger, logs_dir: str = "./logs") -> None:
    """Set up the file logging for the application."""
    os.makedirs(logs_dir, exist_ok=True)

    start_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{logs_dir}/{start_time}.log"
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_formatter.converter = time.gmtime  # use UTC time
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
