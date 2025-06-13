import math
from time import sleep
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Thread
from typing import Union, Optional
from os.path import exists
import pickle
import re

from influxdb_client import InfluxDBClient, QueryApi, WriteApi, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision
from PyQt6.QtCore import QDateTime, QObject, QRunnable, pyqtSignal
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from telcorain.handlers import config_handler, config_handler_out
from telcorain.handlers.logging_handler import logger
from telcorain.procedures.utils.helpers import datetime_rfc
from telcorain.procedures.utils.helpers import measure_time


class BucketType(Enum):
    """
    Enum specifying the type of InfluxDB bucket: 'default' or 'mapped'.
     - In case of 'mapped' bucket, bucket field names are mapped via MariaDB table 'technologies_influx_mapping'.
     - In case of 'default' bucket, default field names are used.
    """

    DEFAULT = "default"
    MAPPED = "mapped"


class InfluxManager:
    """
    InfluxManager class used for communication with InfluxDB database.
    """

    def __init__(self):
        super(InfluxManager, self).__init__()

        # create influx client with parameters from config file
        self.client: InfluxDBClient = InfluxDBClient.from_config_file(
            config_handler.config_path
        )
        self.qapi: QueryApi = self.client.query_api()

        self.options = WriteOptions(
            batch_size=8, flush_interval=8, jitter_interval=0, retry_interval=1000
        )

        # create influx write lock for thread safety (used only when writing output timeseries to InfluxDB)
        self.is_manager_locked = False

        data_border_format = "%Y-%m-%dT%H:%M:%SZ"
        data_border_string = config_handler.read_option(
            "influx2", "old_new_data_border"
        )
        bucket_old_type = getattr(
            BucketType,
            config_handler.read_option("influx2", "old_data_type"),
            BucketType.DEFAULT,
        )
        bucket_new_type = getattr(
            BucketType,
            config_handler.read_option("influx2", "new_data_type"),
            BucketType.DEFAULT,
        )

        self.BUCKET_OLD_DATA: str = config_handler.read_option(
            "influx2", "bucket_old_data"
        )
        self.BUCKET_NEW_DATA: str = config_handler.read_option(
            "influx2", "bucket_new_data"
        )
        self.BUCKET_OUT_CML: str = config_handler_out.read_option(
            "influx2", "bucket_out_cml"
        )
        self.BUCKET_OLD_TYPE: BucketType = bucket_old_type
        self.BUCKET_NEW_TYPE: BucketType = bucket_new_type
        self.OLD_NEW_DATA_BORDER: datetime = datetime.strptime(
            data_border_string, data_border_format
        ).replace(tzinfo=timezone.utc)

    def check_connection(self) -> bool:
        return self.client.ping()

    def _raw_query_old_bucket(
        self, start_str: str, end_str: str, ips_str: str, interval_str: str
    ) -> dict:
        # TODO: needs to be refactored to use the same query for both old and new buckets using BucketType enum
        # construct flux query
        flux = (
            f'from(bucket: "{self.BUCKET_OLD_DATA}")\n'
            + f"  |> range(start: {start_str}, stop: {end_str})\n"
            + f'  |> filter(fn: (r) => r["_field"] == "rx_power" or r["_field"] == "tx_power" or'
            f' r["_field"] == "temperature")\n'
            + f'  |> filter(fn: (r) => r["ip"] =~ /{ips_str}/)\n'
            + f"  |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)\n"
            + f'  |> yield(name: "mean")'
        )

        # query influxDB
        results = self.qapi.query(flux)

        data = {}
        for table in results:
            ip = table.records[0].values.get("ip")

            # initialize new IP record in the result dictionary
            if ip not in data:
                data[ip] = {}
                data[ip]["unit"] = table.records[0].get_measurement()

            # collect data from the current table
            for record in table.records:
                if ip in data:
                    if record.get_field() not in data[ip]:
                        data[ip][record.get_field()] = {}

                    # correct bad Tx Power and Temperature data in InfluxDB in case of missing zero values
                    if (record.get_field() == "tx_power") and (
                        record.get_value() is None
                    ):
                        data[ip]["tx_power"][record.get_time()] = 0.0
                    elif (record.get_field() == "temperature") and (
                        record.get_value() is None
                    ):
                        data[ip]["temperature"][record.get_time()] = 0.0
                    elif (record.get_field() == "rx_power") and (
                        record.get_value() is None
                    ):
                        data[ip]["rx_power"][record.get_time()] = 0.0
                    else:
                        data[ip][record.get_field()][
                            record.get_time()
                        ] = record.get_value()

        return data

    def is_aligned_10_min(self, ts: datetime) -> bool:
        return ts.minute % 10 == 0 and ts.second == 0 and ts.microsecond == 0

    @measure_time
    def _raw_query_new_bucket(
        self, start_str: str, end_str: str, ips_str: str, interval_str: str
    ) -> dict:
        flux = (
            f'from(bucket: "{self.BUCKET_NEW_DATA}")\n'
            + f"  |> range(start: {start_str}, stop: {end_str})\n"
            + f'  |> filter(fn: (r) => r["_field"] == "Teplota" or r["_field"] == "VysilaciVykon" or r["_field"] == "Vysilany_Vykon" or r["_field"] == "PrijimanaUroven" or r["_field"] == "Signal")\n'
            + f'  |> filter(fn: (r) => r["agent_host"] =~ /{ips_str}/)\n'
            + f"  |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)\n"
            + f'  |> yield(name: "mean")'
        )

        try:
            results_flux = self.qapi.query(flux)
        except Exception as e:
            if "compiled too big" in str(e):
                data = self._raw_query_chunks(
                    start_str,
                    end_str,
                    ips_str,
                    interval_str,
                    chunk_size=4000,
                )
                return data

        data = {}

        # map raw fields to the result dictionary
        rename_map = {
            "Teplota": "temperature",
            "PrijimanaUroven": "rx_power",
            "VysilaciVykon": "tx_power",
            "Vysilany_Vykon": "tx_power",
            "Signal": "rx_power",
        }
        for table in results_flux:
            ip = table.records[0].values.get("agent_host")

            # initialize new IP record in the result dictionary
            if ip not in data:
                data[ip] = {}
                data[ip]["unit"] = table.records[0].get_measurement()

            # collect data from the current table
            for record in table.records:
                if ip in data:
                    if not self.is_aligned_10_min(record.get_time()):
                        continue
                    field_name = rename_map.get(record.get_field(), record.get_field())
                    if field_name not in data[ip]:
                        data[ip][field_name] = {}

                    # correct bad Tx Power and Temperature data in InfluxDB in case of missing zero values
                    if (field_name == "tx_power") and (record.get_value() is None):
                        data[ip]["tx_power"][record.get_time()] = 0.0
                    elif (field_name == "temperature") and (record.get_value() is None):
                        data[ip]["temperature"][record.get_time()] = 0.0
                    elif (field_name == "rx_power") and (record.get_value() is None):
                        data[ip]["rx_power"][record.get_time()] = 0.0
                    else:
                        data[ip][field_name][record.get_time()] = record.get_value()
        return data

    @measure_time
    def _raw_query_chunks(
        self,
        start_str: str,
        end_str: str,
        ip_list: list,
        interval_str: str,
        chunk_size: int = 5000,
    ) -> dict:
        """
        Optimized version of the query with better chunk handling.
        """
        all_data = {}
        rename_map = {
            "Teplota": "temperature",
            "PrijimanaUroven": "rx_power",
            "VysilaciVykon": "tx_power",
            "Vysilany_Vykon": "tx_power",
            "Signal": "rx_power",
        }

        # Pre-escape all IPs once
        escaped_ips = [re.escape(ip) for ip in ip_list]

        # Process in chunks but with parallel queries if possible
        for i in range(0, len(escaped_ips), chunk_size):
            ip_chunk = escaped_ips[i : i + chunk_size]
            ips_regex = "|".join(ip_chunk)

            flux = (
                f'from(bucket: "{self.BUCKET_NEW_DATA}")\n'
                f"  |> range(start: {start_str}, stop: {end_str})\n"
                f'  |> filter(fn: (r) => r["_field"] == "Teplota" or '
                f'r["_field"] == "VysilaciVykon" or r["_field"] == "Vysilany_Vykon" or '
                f'r["_field"] == "PrijimanaUroven" or r["_field"] == "Signal")\n'
                f'  |> filter(fn: (r) => r["agent_host"] =~ /{ips_regex}/)\n'
                f"  |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)\n"
                f'  |> yield(name: "mean")'
            )

            try:
                results_flux = self.qapi.query(flux)
                self._process_results(results_flux, all_data, rename_map)
            except Exception as e:
                if "compiled too big" in str(e):
                    self._raw_query_chunks(
                        start_str,
                        end_str,
                        ip_list[i : i + chunk_size],
                        interval_str,
                        chunk_size=chunk_size - 1000,
                    )
                else:
                    raise

        return all_data

    def _process_results(self, results_flux, all_data, rename_map):
        """Helper to process query results."""
        for table in results_flux:
            if not table.records:
                continue

            ip = table.records[0].values.get("agent_host")
            # Only process if we have at least one valid record with a value
            has_valid_data = False

            # First pass: check if we have any valid data for this IP
            for record in table.records:
                if (
                    self.is_aligned_10_min(record.get_time())
                    and record.get_value() is not None
                ):
                    has_valid_data = True
                    break

            if not has_valid_data:
                continue

            # Only now create the IP entry if we have valid data
            if ip not in all_data:
                all_data[ip] = {}
                all_data[ip]["unit"] = table.records[0].get_measurement()

            # Second pass: process the actual data
            for record in table.records:
                if not self.is_aligned_10_min(record.get_time()):
                    continue

                field_name = rename_map.get(record.get_field(), record.get_field())
                if field_name not in all_data[ip]:
                    all_data[ip][field_name] = {}

                value = record.get_value()
                time = record.get_time()

                if value is None and field_name in {
                    "tx_power",
                    "temperature",
                    "rx_power",
                }:
                    all_data[ip][field_name][time] = 0.0
                else:
                    all_data[ip][field_name][time] = value

    @measure_time
    def query_units(
        self,
        ips: list,
        start: datetime,
        end: datetime,
        interval: int,
        rolling_values: int = None,
        compensate_historic: bool = False,
    ) -> dict:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses (as identifiers = tags in InfluxDB).
        Query is done for the time interval defined by 'start' and 'end' QDateTime objects, with 'interval' in seconds.

        :param ips: list of IP addresses of CMLs to query
        :param start: QDateTime object with start of the query interval
        :param end: QDateTime object with end of the query interval
        :param interval: time interval in minutes
        :return: dictionary with queried data, with IP addresses as keys and fields with time series as values
        """

        if compensate_historic:
            num_nan_samples = rolling_values
            compensation_seconds = num_nan_samples * interval * 60
            start -= timedelta(seconds=compensation_seconds)

        # align start down to the nearest `interval` multiple
        start -= timedelta(
            minutes=start.minute % interval,
            seconds=start.second,
            microseconds=start.microsecond,
        )

        end -= timedelta(
            minutes=end.minute % interval,
            seconds=end.second,
            microseconds=end.microsecond,
        )

        # convert params to query substrings
        start_str = datetime_rfc(start)
        end_str = datetime_rfc(end)

        interval_str = f"{interval * 60}s"  # time in seconds

        ips_str = f"{ips[0]}"  # IP addresses in query format
        for ip in ips[1:]:
            ips_str += f"|{ip}"

        return self._raw_query_new_bucket(
            start_str,
            end_str,
            ips_str,
            interval_str,
        )

    def query_units_realtime(
        self,
        ips: list,
        realtime_window_str: str,
        interval: int,
    ) -> dict[str, Union[dict[str, dict[datetime, float]], str]]:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses (as identifiers = tags in InfluxDB).
        Query is done for the time interval defined by 'combo_realtime' QComboBox object.

        :param ips: list of IP addresses of CMLs to query
        :param realtime_window_str: A string describing selected moving time window
        :param interval: time interval in minutes
        :return: dictionary with queried data, with IP addresses as keys and fields with time series as values
        """
        delta_map = {
            "1h": timedelta(hours=1),
            "3h": timedelta(hours=3),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "2d": timedelta(days=2),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        end = datetime.now(timezone.utc)
        start = end - delta_map.get(realtime_window_str)

        return self.query_units(
            ips,
            start,
            end,
            interval,
        )

    def write_points(self, points, bucket):
        with InfluxDBClient.from_config_file(
            config_handler_out.config_path
        ) as client_out:
            try:
                with client_out.write_api() as wapi_out:
                    wapi_out.write(
                        bucket=bucket, record=points, write_precision=WritePrecision.S
                    )
                    sleep(5)
            except Exception as error:
                logger.error(
                    "Error occured during InfluxDB write query, stopping. Error: %s",
                    error,
                )

    @measure_time
    def run_wipeout_output_bucket(self) -> Thread:
        thread = Thread(target=self._wipeout_output_bucket)
        thread.start()
        return thread

    def _wipeout_output_bucket(self):
        attempt = 0
        influx_timeout = int(config_handler_out.read_option("influx2", "timeout"))
        max_attempts = 20 * 1000 / influx_timeout  # wait max 20 seconds
        while True:
            try:
                attempt += 1
                with InfluxDBClient.from_config_file(
                    config_handler_out.config_path
                ) as client_out:
                    client_out.delete_api().delete(
                        start="1970-01-01T00:00:00Z",
                        stop="2100-01-01T00:00:00Z",
                        predicate="",
                        bucket=self.BUCKET_OUT_CML,
                    )
                logger.debug(
                    "[DEVMODE] InfluxDB outputs wipeout successful after %d attempts.",
                    attempt,
                )
                logger.info("[DEVMODE] InfluxDB output bucket erased.")
                break
            except (ConnectTimeoutError, ReadTimeoutError):
                logger.debug(
                    "[DEVMODE] Timeout during InfluxDB outputs wipeout. "
                    "Attempt #%d. Calling again...",
                    attempt,
                )
                if attempt > max_attempts:
                    logger.error(
                        "[DEVMODE] Timeout during InfluxDB outputs wipeout. Maximum attempts reached. "
                        "Check state of bucket %s. Skipping.",
                        self.BUCKET_OUT_CML,
                    )
                    break
                continue


class InfluxChecker(InfluxManager, QRunnable):
    """
    InfluxChecker class for connection testing with InfluxDB database. Use in threadpool.
    Emits 'ping_signal' from 'InfluxSignal' class passed as 'signals' parameter.
    """

    def __init__(self, signals: QObject):
        super(InfluxChecker, self).__init__()
        self.sig = signals

    def run(self):
        self.sig.ping_signal.emit(self.check_connection())


class InfluxSignals(QObject):
    """
    InfluxSignals class for signaling between InfluxManager and its threadpooled subclasses.
    """

    ping_signal = pyqtSignal(bool)


# global instance of InfluxManager, accessible from all modules
influx_man = InfluxManager()
