import logging
import os
import sys
import time
from datetime import datetime
from io import TextIOWrapper
import configparser
from os.path import exists


# =====================================================================
# Config handling
# =====================================================================


class ConfigHandler:
    """Class for handling and reading configuration file."""

    def __init__(self):
        self.config_path = "configs/config.ini"

        self.configs = configparser.ConfigParser()
        self.sections = []

        if exists(self.config_path):
            self.configs.read(self.config_path, encoding="utf-8")
            self.sections = self.configs.sections()
        else:
            raise FileNotFoundError(
                "Missing configuration file! Check the config.ini file in root directory."
            )

    def read_option(self, section: str, option: str) -> str:
        """
        Reads a specific option from a specific section in the configuration file.
        """
        if not self.configs.has_option(section, option):
            raise ConfigOptionNotFound(section, option)
        return self.configs[section][option]

    def load_sql_config(self) -> dict[str, str]:
        """
        Loads the MariaDB configuration from the configuration file.
        """
        sql_configs = {
            "address": self.read_option("mariadb", "address"),
            "port": self.read_option("mariadb", "port"),
            "user": self.read_option("mariadb", "user"),
            "pass": self.read_option("mariadb", "pass"),
            "timeout": self.read_option("mariadb", "timeout"),
            "db_metadata": self.read_option("mariadb", "db_metadata"),
            "db_output": self.read_option("mariadb", "db_output"),
            "exclude_cmls_path": self.read_option("mariadb", "exclude_cmls_path"),
        }

        return sql_configs


class ConfigOptionNotFound(Exception):
    def __init__(self, section: str, option: str):
        self.section = section
        self.option = option
        super().__init__(
            f"Missing option in configuration file. Check the config! "
            f"Section: {section}, Option: {option}"
        )


# =====================================================================
# Logging
# =====================================================================


class InitLogHandler(logging.Handler):
    """
    Logging handler that prints to stdout/stderr and can buffer records
    (buffer is kept in case you ever want to inspect it later).
    """

    def __init__(self):
        super().__init__()
        self.buffer: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)
        msg = self.format(record)
        # DEBUG/INFO/WARNING -> stdout; ERROR/CRITICAL -> stderr
        if record.levelno < logging.ERROR:
            print(msg, file=sys.stdout)
        else:
            print(msg, file=sys.stderr)


def setup_logging(
    logger: logging.Logger,
    config_handler: ConfigHandler,
    logs_dir: str = "./logs",
) -> None:
    """
    Configure logging for the whole application:

    - console output via InitLogHandler (stdout/stderr split)
    - file output with UTC timestamps
    - logger level from [logging] init_level in config.ini
    """

    # Avoid attaching duplicate handlers if called multiple times
    if logger.handlers:
        return

    # Ensure UTF-8, line-buffered stdout
    sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    # Level from config
    level_name = config_handler.read_option("logging", "init_level")
    logger.setLevel(level_name)

    # ------------------------------------------------------------------
    # Console handler (InitLogHandler)
    # ------------------------------------------------------------------
    console_handler = InitLogHandler()
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_formatter.converter = time.gmtime  # UTC
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler
    # ------------------------------------------------------------------
    os.makedirs(logs_dir, exist_ok=True)
    start_time = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(logs_dir, f"{start_time}.log")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter.converter = time.gmtime  # UTC
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# =====================================================================
# Global instances
# =====================================================================

config_handler = ConfigHandler()
logger = logging.getLogger("telcorain")
setup_logging(
    logger,
    config_handler,
    logs_dir=config_handler.read_option("directories", "logs"),
)
