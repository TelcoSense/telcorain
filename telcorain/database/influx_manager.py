import math
from time import sleep
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Thread
from typing import Union, Optional
from os.path import exists
import pickle

from influxdb_client import InfluxDBClient, QueryApi, WriteApi, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision
from PyQt6.QtCore import QDateTime, QObject, QRunnable, pyqtSignal
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from telcorain.handlers import config_handler, config_handler_out
from telcorain.handlers.logging_handler import logger
from telcorain.procedures.utils.helpers import datetime_rfc
from telcorain.procedures.utils.helpers import measure_time


@measure_time
def get_max_min_time(
    data: dict[str, Union[dict[str, dict[datetime, float]], str]]
) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Gets the maximum and minimum time values from the data.

    :param data: The input data structure with timestamps as datetime objects.
    :return: A tuple containing the max and min time values as datetime objects, or (None, None) if no valid timestamps are found.
    """
    all_times = []

    # Traverse the data structure to collect timestamps
    for key, value in data.items():
        if isinstance(value, dict):  # Only process nested dictionaries
            for nested_key, time_value_dict in value.items():
                if isinstance(time_value_dict, dict):
                    # Collect all datetime keys
                    all_times.extend(time_value_dict.keys())

    # Check if any timestamps were found
    if not all_times:
        return None, None

    # Get min and max times directly as datetime objects
    min_time = min(all_times)
    max_time = max(all_times)

    return min_time, max_time


def count_entries(data: dict[str, Union[dict, str]]) -> int:
    """
    Recursively counts the number of entries in a nested dictionary.

    :param data: the dictionary (influx_data) to count entries for.
    :return: the total number of entries.
    """
    count = 0
    for value in data.values():
        if isinstance(value, dict):
            # recursively count nested dictionaries
            count += count_entries(value)
        elif isinstance(value, str):
            # strings are metadata, count as one
            count += 1
    return count


@measure_time
def filter_and_prepend(
    data: dict[str, Union[dict[str, dict[datetime, float]], str]],
    new_data: dict[str, dict[datetime, float]],
    first_entry_time: datetime,
) -> dict[str, Union[dict[str, dict[datetime, float]], str]]:
    """
    Update old data with new data while ensuring the keys of both datasets remain the same.

    :param data: Old dataset from the previous run.
    :param new_data: New dataset fetched in the current run.
    :param current_time: The current timestamp.
    :param retention_window: The time window for filtering old data.
    :return: Updated dataset with filtered and merged data.
    """
    updated_data = {}

    # Step 1: Find common keys
    old_keys = set(data.keys())
    new_keys = set(new_data.keys())
    common_keys = old_keys & new_keys  # Intersection of old and new keys

    # print(f"Keys common to both: {common_keys}")
    print(f"Keys only in old_data (will be removed): {old_keys - common_keys}")
    print(f"Keys only in new_data (will be removed): {new_keys - common_keys}")

    # Step 2: Filter old_data to retain common keys and filter timestamps
    for key in common_keys:
        if isinstance(data[key], dict):
            filtered_nested_dict = {}
            for nested_key, time_value_dict in data[key].items():
                if isinstance(time_value_dict, dict):
                    # Filter out old timestamps
                    filtered_time_value_dict = {
                        time: val
                        for time, val in time_value_dict.items()
                        if time > first_entry_time
                    }
                    if filtered_time_value_dict:
                        filtered_nested_dict[nested_key] = filtered_time_value_dict
                    # print(
                    #     f"Key: {key}, Nested Key: {nested_key}, Filtered Size: {len(filtered_time_value_dict)}"
                    # )
            updated_data[key] = filtered_nested_dict
        else:
            updated_data[key] = data[key]  # Preserve non-dict values

    # Step 3: Merge new_data into updated_data for common keys
    for key in common_keys:
        for nested_key, new_time_value_dict in new_data[key].items():
            if nested_key not in updated_data[key]:
                updated_data[key][nested_key] = new_time_value_dict
            else:
                # Avoid duplicating timestamps
                existing_times = set(updated_data[key][nested_key].keys())
                new_entries = {
                    time: val
                    for time, val in new_time_value_dict.items()
                    if time not in existing_times
                }
                updated_data[key][nested_key].update(new_entries)
                # print(
                #     f"Key: {key}, Nested Key: {nested_key}, New Entries Added: {len(new_entries)}"
                # )

    # Debug final size
    final_size = sum(
        sum(len(time_dict) for time_dict in value.values())
        for key, value in updated_data.items()
        if isinstance(value, dict)
    )
    print(f"Len of new_data: {len(data)}")
    print(f"Final size of updated data: {final_size}")
    return updated_data


@measure_time
def filter_and_prepend2(
    data: dict[str, Union[dict[str, dict[datetime, float]], str]],
    new_data: dict[str, dict[datetime, float]],
    first_entry_time: datetime,
    handle_missing: str = "filter",  # options: "filter" or "append"
) -> dict[str, Union[dict[str, dict[datetime, float]], str]]:
    """
    Update old_influx_data with new_influx_data, handling missing keys (measurements) based on the specified strategy.
    This is needed for mergind old and new influx data if the dimensions are not consistent.

    :param data: old data from the previous run
    :param new_data: new data within this run
    :param first_entry_time: time of the end of old data (first entry of new data)
    :param handle_missing: strategy for missing keys. "filter" to discard, "append" to preserve data
    :return: updated dataset with filtered and merged data
    """
    updated_data = {}

    # filter old data based on retention window given by first_entry_time
    for key, value in data.items():
        if isinstance(value, dict):
            filtered_nested_dict = {}
            for nested_key, time_value_dict in value.items():
                if isinstance(time_value_dict, dict):
                    # keep only entries within the retention window
                    filtered_nested_dict[nested_key] = {
                        time: val
                        for time, val in time_value_dict.items()
                        if time > first_entry_time
                    }
            updated_data[key] = filtered_nested_dict
        else:
            updated_data[key] = value  # keep non-dictionary values (e.g., metadata)

    # handle keys missing from new_data
    old_keys = set(data.keys())
    new_keys = set(new_data.keys())

    if handle_missing == "filter":
        # remove keys from old_data that are missing in new_data
        updated_data = {
            key: updated_data[key] for key in updated_data if key in new_keys
        }
    elif handle_missing == "append":
        # preserve old_data keys that are missing in new_data
        missing_keys = old_keys - new_keys
        for key in missing_keys:
            updated_data[key] = data[key]

    # merge new data into updated_data
    for key, time_value_dict in new_data.items():
        if key not in updated_data:
            # add entirely new keys
            updated_data[key] = {"default": time_value_dict}
        elif isinstance(updated_data[key], dict):
            for nested_key, new_time_value_dict in time_value_dict.items():
                if nested_key not in updated_data[key]:
                    # add new nested keys
                    updated_data[key][nested_key] = new_time_value_dict
                else:
                    # update existing nested keys with newer timestamps
                    most_recent_time = max(
                        updated_data[key][nested_key].keys(), default=first_entry_time
                    )
                    updated_data[key][nested_key].update(
                        {
                            time: val
                            for time, val in new_time_value_dict.items()
                            if time > most_recent_time
                        }
                    )
    return updated_data


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

    @measure_time
    def _raw_query_new_bucket(
        self, start_str: str, end_str: str, ips_str: str, interval_str: str
    ) -> dict:
        # TODO: needs to be refactored to use the same query for both old and new buckets using BucketType enum
        # construct flux query
        slux = (
            f'from(bucket: "{self.BUCKET_NEW_DATA}")\n'
            + f"  |> range(start: {start_str}, stop: {end_str})\n"
            + f'  |> filter(fn: (r) => r["_field"] == "PrijimanaUroven" or r["_field"] == "Signal")\n'
            + f'  |> filter(fn: (r) => r["agent_host"] =~ /{ips_str}/)\n'
            + f"  |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)\n"
            + f'  |> yield(name: "mean")'
        )

        flux = (
            f'from(bucket: "{self.BUCKET_NEW_DATA}")\n'
            + f"  |> range(start: {start_str}, stop: {end_str})\n"
            + f'  |> filter(fn: (r) => r["_field"] == "Teplota" or r["_field"] == "VysilaciVykon" or r["_field"] == "Vysilany_Vykon")\n'
            + f'  |> filter(fn: (r) => r["agent_host"] =~ /{ips_str}/)\n'
            + f"  |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)\n"
            + f'  |> yield(name: "mean")'
        )

        # query influxDB
        results_slux = self.qapi.query(slux)
        results_flux = self.qapi.query(flux)

        data = {}

        # map raw fields to the result dictionary
        rename_map = {
            "Teplota": "temperature",
            "PrijimanaUroven": "rx_power",
            "VysilaciVykon": "tx_power",
            "Vysilany_Vykon": "tx_power",
            "Signal": "rx_power",
        }
        for results in (results_slux, results_flux):
            for table in results:
                ip = table.records[0].values.get("agent_host")

                # initialize new IP record in the result dictionary
                if ip not in data:
                    data[ip] = {}
                    data[ip]["unit"] = table.records[0].get_measurement()

                # collect data from the current table
                for record in table.records:
                    if ip in data:
                        field_name = rename_map.get(
                            record.get_field(), record.get_field()
                        )
                        if field_name not in data[ip]:
                            data[ip][field_name] = {}

                        # correct bad Tx Power and Temperature data in InfluxDB in case of missing zero values
                        if (field_name == "tx_power") and (record.get_value() is None):
                            data[ip]["tx_power"][record.get_time()] = 0.0
                        elif (field_name == "temperature") and (
                            record.get_value() is None
                        ):
                            data[ip]["temperature"][record.get_time()] = 0.0
                        elif (field_name == "rx_power") and (
                            record.get_value() is None
                        ):
                            data[ip]["rx_power"][record.get_time()] = 0.0
                        else:
                            data[ip][field_name][record.get_time()] = record.get_value()
        return data

    @measure_time
    def query_units(
        self,
        ips: list,
        start: datetime,
        end: datetime,
        interval: int,
        realtime_optimization: bool = False,
        force_data_refresh: bool = False,
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
        orig_start = start
        if realtime_optimization and not force_data_refresh:
            old_influx_data = None
            if exists("temp_data/temp_data.pkl"):
                with open("temp_data/temp_data.pkl", "rb") as f:
                    old_influx_data = pickle.load(f)
                # get the min and max (start and end) of previous influx data download iteration
                _, old_end = get_max_min_time(old_influx_data)
                # if the data are recent, new start equals end of previous iteration
                if old_end > start:
                    start = old_end

        # modify boundary times to be multiples of input time interval
        start += timedelta(
            seconds=(
                (math.ceil((start.minute + 0.1) / interval) * interval) - start.minute
            )
            * 60
        )
        end += timedelta(seconds=(-1 * (end.minute % interval)) * 60)

        # convert params to query substrings
        start_str = datetime_rfc(start)
        end_str = datetime_rfc(end)

        interval_str = f"{interval * 60}s"  # time in seconds

        ips_str = f"{ips[0]}"  # IP addresses in query format
        for ip in ips[1:]:
            ips_str += f"|{ip}"

        if end < self.OLD_NEW_DATA_BORDER:
            return self._raw_query_old_bucket(start_str, end_str, ips_str, interval_str)
        else:
            if (
                realtime_optimization
                and old_influx_data is not None
                and not force_data_refresh
            ):
                logger.info(
                    "[INFO] Updating the start and end of new data query to: from %s to %s.",
                    old_end,
                    end,
                )
                new_influx_data = self._raw_query_new_bucket(
                    start_str, end_str, ips_str, interval_str
                )

                len_of_old_data = count_entries(old_influx_data)
                len_of_new_data = count_entries(new_influx_data)

                if len_of_old_data != len_of_new_data:
                    logger.info(
                        "[INFO] The number of CMLs for old_influx_data and new_influx_data is different! (%d vs. %d)",
                        len_of_old_data,
                        len_of_new_data,
                    )

                # # Check difference between old_influx_data and new_influx_data
                # old_keys = set(old_influx_data.keys())
                # new_keys = set(new_influx_data.keys())

                # # Compare top-level keys
                # missing_in_new = old_keys - new_keys
                # missing_in_old = new_keys - old_keys

                # print(f"Keys in old_data but not in new_data: {missing_in_new}")
                # print(f"Keys in new_data but not in old_data: {missing_in_old}")
                updated_data = filter_and_prepend(
                    old_influx_data, new_influx_data, orig_start
                )
                print(
                    f"Updated data size: {sum(len(v) for k, v in updated_data.items() if isinstance(v, dict))}"
                )
                return updated_data
            else:
                return self._raw_query_new_bucket(
                    start_str, end_str, ips_str, interval_str
                )

    def query_units_realtime(
        self,
        ips: list,
        realtime_window_str: str,
        interval: int,
        realtime_optimization: bool = False,
        force_data_refresh: bool = False,
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
            realtime_optimization=realtime_optimization,
            force_data_refresh=force_data_refresh,
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
