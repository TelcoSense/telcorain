"""Module containing class for handling MariaDB connection."""

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
    Class for handling MariaDB connection and database data loading/writing.
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

        # current realtime params DB ID
        self.realtime_params_id = 0

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

    def get_link_ids_by_ips(
        self,
        txt_file_path: str,
    ) -> list[int]:
        """
        Load link IDs from MariaDB for the IPs listed in a given text file.

        :param txt_file_path: Path to the text file containing one IP per line.
        :param connection: Active MariaDB connection object.
        :return: List of matching link IDs.
        """
        try:
            # Step 1: Read IPs from file
            with open(txt_file_path, "r") as file:
                ip_list = [line.strip() for line in file if line.strip()]

            if not ip_list:
                logger.warning("No IPs found in the provided file.")
                return []

            cursor = self.connection.cursor()

            # Step 2: Prepare and run SQL query
            placeholder = ", ".join(["%s"] * len(ip_list))
            query = f"""
            SELECT ID FROM links
            WHERE IP_address_A IN ({placeholder})
            OR IP_address_B IN ({placeholder});
            """

            # Provide the IPs twice (for IP_address_A and IP_address_B)
            cursor.execute(query, ip_list + ip_list)

            # Step 3: Collect matching link IDs
            matching_ids = [row[0] for row in cursor.fetchall()]

            return matching_ids

        except mariadb.Error as e:
            logger.error("Failed to fetch link IDs by IPs: %s", e)
            return []
        except FileNotFoundError:
            logger.error("File not found: %s", txt_file_path)
            return []

    def load_metadata(
        self,
        ids=None,
        min_length: int = 0.01,
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

                if exclude_ids:
                    with open(str(self.settings["exclude_cmls_path"]), newline="") as f:
                        reader = csv.reader(f)
                        invalid_ids = set(int(row[0]) for row in reader if row)

                links = {}

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

    def get_last_realtime(self) -> dict[str, Union[str, int, float, datetime]]:
        """
        Get parameters of last running realtime calculation from output database.

        :return: Dictionary of realtime parameters. Key is parameter name, value is parameter value.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    "SELECT started, retention, timestep, resolution, X_MIN, X_MAX, Y_MIN, Y_MAX "
                    f"FROM {self.settings['db_output']}.realtime_rain_parameters "
                    "ORDER BY started DESC "
                    "LIMIT 1;"
                )

                cursor.execute(query)

                realtime_params = {}

                for (
                    started,
                    retention,
                    timestep,
                    resolution,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                ) in cursor:
                    realtime_params = {
                        "start_time": started,
                        "retention": retention,
                        "timestep": timestep,
                        "resolution": resolution,
                        "X_MIN": X_MIN,
                        "X_MAX": X_MAX,
                        "Y_MIN": Y_MIN,
                        "Y_MAX": Y_MAX,
                    }

                return realtime_params
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)
            return {}

    def get_realtime(self, parameters_id: id) -> dict[str, Union[int, float, datetime]]:
        """
        Get parameters of specific realtime calculation from output database.

        :param parameters_id: ID of the realtime parameters.
        :return: Dictionary of realtime parameters. Key is parameter name, value is parameter value.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    "SELECT started, retention, timestep, resolution, X_MIN, X_MAX, Y_MIN, Y_MAX, "
                    f"X_count, Y_count FROM {self.settings['db_output']}.realtime_rain_parameters "
                    "WHERE ID = ?;"
                )

                cursor.execute(query, (parameters_id,))

                realtime_params = {}

                for (
                    started,
                    retention,
                    timestep,
                    resolution,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    X_count,
                    Y_count,
                ) in cursor:
                    realtime_params = {
                        "start_time": started,
                        "retention": retention,
                        "timestep": timestep,
                        "resolution": float(resolution),
                        "X_MIN": float(X_MIN),
                        "X_MAX": float(X_MAX),
                        "Y_MIN": float(Y_MIN),
                        "Y_MAX": float(Y_MAX),
                        "X_count": X_count,
                        "Y_count": Y_count,
                    }

                return realtime_params
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)
            return {}

    def insert_realtime(
        self,
        retention: int,
        timestep: int,
        resolution: float,
        X_MIN: float,
        X_MAX: float,
        Y_MIN: float,
        Y_MAX: float,
    ):
        """
        Insert realtime parameters into output database.

        :param retention: Retention time in minutes.
        :param timestep: Timestep in seconds.
        :param resolution: Resolution in decimal degrees.
        :param X_MIN: Minimum longitude.
        :param X_MAX: Maximum longitude.
        :param Y_MIN: Minimum latitude.
        :param Y_MAX: Maximum latitude.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    f"INSERT INTO {self.settings['db_output']}.realtime_rain_parameters "
                    "(retention, timestep, resolution, X_MIN, X_MAX, Y_MIN, Y_MAX, X_count, Y_count, images_URL)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
                )

                x = int((X_MAX - X_MIN) / resolution + 1)
                y = int((Y_MAX - Y_MIN) / resolution + 1)

                cursor.execute(
                    query,
                    (
                        retention,
                        timestep,
                        resolution,
                        X_MIN,
                        X_MAX,
                        Y_MIN,
                        Y_MAX,
                        x,
                        y,
                        "",
                    ),
                )
                self.connection.commit()

                # store the ID of the inserted record
                self.realtime_params_id = cursor.lastrowid
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to insert data into MariaDB: %s", e)

    def get_last_raingrid(self) -> dict[datetime, list[int]]:
        """
        Get last raingrid from output database.

        :return: Dictionary of last raingrid. Key is time, value is list of CML IDs.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    f"SELECT time, links FROM {self.settings['db_output']}.realtime_rain_grids "
                    f"ORDER BY time DESC LIMIT 1;"
                )

                cursor.execute(query)

                last_raingrid = {}

                for time, links in cursor:
                    last_raingrid[time] = json.loads(links)

                return last_raingrid
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)
            return {}

    def verify_raingrid(self, parameters: id, time: datetime) -> bool:
        """
        Verify if raingrid with given parameters and time already exists in output database.

        :param parameters: ID of the realtime parameters.
        :param time: Time of the raingrid.
        :return: True if raingrid exists, False otherwise.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    f"SELECT COUNT(*) FROM {self.settings['db_output']}.realtime_rain_grids "
                    f"WHERE time = ? AND parameters = ?;"
                )

                cursor.execute(query, (time, parameters))

                count = cursor.fetchone()[0]

                return count > 0
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to read data from MariaDB: %s", e)

    def insert_raingrid(
        self,
        time: datetime,
        links: list[int],
        file_name: str,
        r_median: float,
        r_avg: float,
        r_max: float,
    ):
        """
        Insert raingrid's metadata into output database.

        :param time: Time of the raingrid.
        :param links: List of CML IDs.
        :param file_name: Name of the generated raingrid SVG image file.
        :param r_median: Median rain intensity value in given raingrid.
        :param r_avg: Average rain intensity value in given raingrid.
        :param r_max: Maximum rain intensity value in given raingrid.
        """
        if self.realtime_params_id == 0:
            raise ValueError(
                "Unknown parameters ID. Realtime parameters has not been set?"
            )

        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                query = (
                    f"INSERT INTO {self.settings['db_output']}.realtime_rain_grids "
                    f"(time, parameters, links, image_name, R_MEDIAN, R_AVG, R_MAX) VALUES (?, ?, ?, ?, ?, ?, ?);"
                )

                cursor.execute(
                    statement=query,
                    data=(
                        time,
                        self.realtime_params_id,
                        json.dumps(links),
                        file_name,
                        r_median,
                        r_avg,
                        r_max,
                    ),
                )
                self.connection.commit()
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to insert data into MariaDB: %s", e)

    def delete_old_data(self, current_time: datetime, retention_window: datetime):
        """
        Delete data that are older than retention_window
        """
        total_delete_rows = 0
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()
                # Calculate the threshold
                threshold = (current_time - retention_window).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                # SQL query to delete rows older than the threshold
                query1 = f"""
                DELETE FROM {self.settings['db_output']}.realtime_rain_grids WHERE time < %s
                """
                query2 = f"""
                DELETE FROM {self.settings['db_output']}.realtime_rain_parameters WHERE started < %s
                """
                cursor.execute(query1, (threshold,))
                total_delete_rows += cursor.rowcount
                cursor.execute(query2, (threshold,))
                total_delete_rows += cursor.rowcount

                self.connection.commit()
        except mariadb.Error as e:
            logger.error("Failed to delete data in MariaDB: %s", e)
        return total_delete_rows

    def wipeout_realtime_tables(self):
        """
        Truncate realtime tables in output database.
        """
        try:
            if self.check_connection():
                cursor: Cursor = self.connection.cursor()

                queries = (
                    "SET FOREIGN_KEY_CHECKS = 0;",
                    f"TRUNCATE TABLE {self.settings['db_output']}.realtime_rain_grids;",
                    f"TRUNCATE TABLE {self.settings['db_output']}.realtime_rain_parameters;",
                    "SET FOREIGN_KEY_CHECKS = 1;",
                )

                for query in queries:
                    cursor.execute(query)
                self.connection.commit()
            else:
                raise mariadb.Error("Connection is not active.")
        except mariadb.Error as e:
            logger.error("Failed to insert data into MariaDB: %s", e)
        else:
            logger.info("[DEVMODE] MariaDB output tables erased.")

    def __del__(self):
        self.connection.close()
