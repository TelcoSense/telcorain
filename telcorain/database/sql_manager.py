"""Module containing class for handling MariaDB connection (read-only)."""

import json
import csv
from datetime import datetime
from typing import Union

import mariadb
from mariadb import Cursor

from telcorain.handlers import config_handler, logger
from telcorain.helpers import calc_distance, MwLink


class SqlManager:
    """
    Class for handling MariaDB connection and database data loading.

    This version is read-only:
      - Only metadata and (optionally) last_raingrid are read.
      - No INSERT / TRUNCATE / UPDATE statements are executed.
    """

    # Do not spam log with error messages
    is_error_sent = False

    def __init__(self):
        super(SqlManager, self).__init__()
        # Load settings from config file via ConfigurationManager
        self.settings = config_handler.load_sql_config()
        # Init empty connections
        self.connection = None
        # Define connection state
        self.is_connected = False

    def connect(self):
        """
        Connect to MariaDB database.
        """
        try:
            self.connection = mariadb.connect(
                user=self.settings["user"],
                password=self.settings["pass"],
                host=self.settings["address"],
                port=int(self.settings["port"]),
                database=self.settings["db_metadata"],
                connect_timeout=int(int(self.settings["timeout"]) / 1000),
                reconnect=True,
            )

            self.is_connected = True
            SqlManager.is_error_sent = False

        except mariadb.Error as e:
            if not SqlManager.is_error_sent:
                logger.error("Cannot connect to MariaDB Platform: %s", e)
                SqlManager.is_error_sent = True
            self.is_connected = False

    def check_connection(self) -> bool:
        """
        Check connection state if it is still active.

        :return: True if connection is active, False otherwise.
        """
        if self.is_connected:
            try:
                self.connection.ping()
                return True
            except (mariadb.InterfaceError, mariadb.OperationalError):
                return False
        else:
            self.connect()
            return self.is_connected

    def load_metadata(
        self,
        ids=None,
        min_length: float = 0.01,
        max_length: float = float("inf"),
        exclude_ids: bool = True,
    ) -> dict[int, MwLink]:
        """
        Load metadata of CMLs from MariaDB.

        :param ids: Optional list of link IDs to load.
        :param min_length: Minimum allowed link length (in km).
        :param max_length: Maximum allowed link length (in km).
        :return: Dictionary of CMLs metadata. Key is CML ID, value is MwLink model object.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = """
                SELECT
                  links.ID,
                  links.IP_address_A,
                  links.IP_address_B,
                  links.frequency_A,
                  links.frequency_B,
                  links.polarization,
                  sites_A.address AS address_A,
                  sites_B.address AS address_B,
                  sites_A.X_coordinate AS longitude_A,
                  sites_B.X_coordinate AS longitude_B,
                  sites_A.Y_coordinate AS latitude_A,
                  sites_B.Y_coordinate AS latitude_B,
                  sites_A.X_dummy_coordinate AS dummy_longitude_A,
                  sites_B.X_dummy_coordinate AS dummy_longitude_B,
                  sites_A.Y_dummy_coordinate AS dummy_latitude_A,
                  sites_B.Y_dummy_coordinate AS dummy_latitude_B,
                  technologies.name AS technology_name,
                  technologies_influx_mapping.measurement AS technology_influx
                FROM
                  links
                JOIN sites AS sites_A ON links.site_A = sites_A.ID
                JOIN sites AS sites_B ON links.site_B = sites_B.ID
                JOIN technologies ON links.technology = technologies.ID
                JOIN technologies_influx_mapping ON technologies.influx_mapping_ID = technologies_influx_mapping.ID
                """

                # filtering based on IDs of the links if provided
                if ids is not None and len(ids) > 0:
                    placeholder = ", ".join(["%s"] * len(ids))
                    query += f" WHERE links.ID IN ({placeholder})"
                    query += ";"
                    cursor.execute(query, ids)
                else:
                    cursor.execute(query)

                invalid_ids = set()
                if exclude_ids:
                    with open(str(self.settings["exclude_cmls_path"]), newline="") as f:
                        reader = csv.reader(f)
                        invalid_ids = {int(row[0]) for row in reader if row}

                links: dict[int, MwLink] = {}

                for (
                    ID,
                    IP_address_A,
                    IP_address_B,
                    frequency_A,
                    frequency_B,
                    polarization,
                    address_A,
                    address_B,
                    longitude_A,
                    longitude_B,
                    latitude_A,
                    latitude_B,
                    dummy_longitude_A,
                    dummy_longitude_B,
                    dummy_latitude_A,
                    dummy_latitude_B,
                    technology_name,
                    technology_influx,
                ) in cursor:

                    link_length = calc_distance(
                        latitude_A, longitude_A, latitude_B, longitude_B
                    )

                    if link_length < min_length or link_length > max_length:
                        continue

                    if exclude_ids and ID in invalid_ids:
                        continue

                    link = MwLink(
                        link_id=ID,
                        name=address_A + " <-> " + address_B,
                        tech=technology_influx,
                        name_a=address_A,
                        name_b=address_B,
                        freq_a=frequency_A,
                        freq_b=frequency_B,
                        polarization=polarization,
                        ip_a=IP_address_A,
                        ip_b=IP_address_B,
                        distance=link_length,
                        latitude_a=latitude_A,
                        longitude_a=longitude_A,
                        latitude_b=latitude_B,
                        longitude_b=longitude_B,
                        dummy_latitude_a=dummy_latitude_A,
                        dummy_longitude_a=dummy_longitude_A,
                        dummy_latitude_b=dummy_latitude_B,
                        dummy_longitude_b=dummy_longitude_B,
                    )

                    links[ID] = link

                return links
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)
            return {}

    def get_last_raingrid(self) -> dict[datetime, list[int]]:
        """
        Get last raingrid from output database.

        NOTE: This is kept for compatibility but is not used by Writer anymore.
        You can delete this method if nothing else in your codebase needs it.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    f"SELECT time, links FROM {self.settings['db_output']}.realtime_rain_grids "
                    f"ORDER BY time DESC LIMIT 1;"
                )

                cursor.execute(query)

                last_raingrid: dict[datetime, list[int]] = {}

                for time, links in cursor:
                    last_raingrid[time] = json.loads(links)

                return last_raingrid
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)
            return {}

    def __del__(self):
        try:
            if self.connection is not None:
                self.connection.close()
        except Exception:
            pass
