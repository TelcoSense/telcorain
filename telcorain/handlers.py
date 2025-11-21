import logging
import os
import sys
import time
from datetime import datetime
from io import TextIOWrapper
import configparser
from os.path import exists
import codecs


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

        :param section: The section in the configuration file.
        :param option: The option in the section.
        :return: The value of the option.

        :raises ConfigOptionNotFound: If the option is not found in the configuration file.
        """
        if not self.configs.has_option(section, option):
            raise ConfigOptionNotFound(section, option)
        return self.configs[section][option]

    def load_sql_config(self) -> dict[str, str]:
        """
        Loads the MariaDB configuration from the configuration file.
        :return: A dictionary containing the MariaDB configuration.
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
            f"Missing option in configuration file. Check the config! Section: {section}, Option: {option}"
        )


class LinksetsHandler:
    def __init__(self, links: dict):
        self.links = links
        self.sets_path = "./linksets.ini"

        self.linksets = configparser.ConfigParser()
        self.sections = []

        if exists(self.sets_path):
            self.linksets.read(self.sets_path, encoding="utf-8")
            self.sections = self.linksets.sections()

        # ////// SQL DB -> ini file synchronization \\\\\\

        # check listed links in link sets and remove old/deleted/invalid links
        if len(self.linksets["DEFAULT"]) > 0:
            links_for_del = []

            for link_id in self.linksets["DEFAULT"]:
                if int(link_id) not in self.links:
                    links_for_del.append(link_id)

            if len(links_for_del) > 0:
                for link_id in links_for_del:
                    self.linksets["DEFAULT"].pop(link_id)

                for link_set in self.sections:
                    for link_id in links_for_del:
                        try:
                            self.linksets[link_set].pop(link_id)
                        except KeyError:
                            logger.warning(
                                f"Deleted link ID {link_id} not found in link set {link_set}."
                            )

        # add new/missing links into linksets default list
        for link_id in self.links:
            if str(link_id) not in self.linksets["DEFAULT"]:
                self.linksets["DEFAULT"][str(link_id)] = "3"

        # save changes into ini
        self.save()

    def create_set(self, name: str):
        self.linksets[name] = {}
        self.sections.append(name)

        for link_id in self.linksets["DEFAULT"]:
            self.linksets[name][link_id] = "0"

        self.save()

    def copy_set(self, origin_name: str, new_name: str):
        self.linksets[new_name] = {}
        self.sections.append(new_name)

        for link_id in self.linksets[origin_name]:
            if self.linksets[origin_name][link_id] != 3:
                self.linksets[new_name][link_id] = self.linksets[origin_name][link_id]

        self.save()

    def delete_set(self, name: str):
        self.linksets.remove_section(name)
        self.save()

    def modify_link(self, set_name: str, link_id: int, channels: int):
        self.linksets[set_name][str(link_id)] = str(channels)

    def delete_link(self, set_name: str, link_id: int):
        self.linksets.remove_option(set_name, str(link_id))

    def save(self):
        with codecs.open(self.sets_path, "w", "utf-8") as setsfile:
            self.linksets.write(setsfile)


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
    # stream_handler = logging.StreamHandler(sys.stdout)
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
    # stream_handler.setFormatter(file_formatter)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


logger = logging.getLogger("telcorain")
config_handler = ConfigHandler()
