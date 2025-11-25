from time import sleep
from datetime import datetime, timedelta, timezone
from typing import List

import re
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient, QueryApi, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision

from telcorain.handlers import config_handler, logger
from telcorain.helpers import datetime_rfc, measure_time


class InfluxManager:
    """
    InfluxManager class used for communication with InfluxDB database.

    - Uses Flux aggregateWindow + pivot on the server side.
    - Fetches results via query_data_frame into pandas.
    - Returns a single wide DataFrame:

        _time (datetime64[ns, UTC])
        agent_host (str)
        temperature (float)
        rx_power (float)
        tx_power (float)

      where temperature / rx_power / tx_power are already mapped from:
        Teplota, PrijimanaUroven / Signal, VysilaciVykon / Vysilany_Vykon.
    """

    def __init__(self):
        super(InfluxManager, self).__init__()

        # create influx client with parameters from config file
        self.client: InfluxDBClient = InfluxDBClient.from_config_file(
            config_handler.config_path
        )
        self.qapi: QueryApi = self.client.query_api()

        self.options = WriteOptions(
            batch_size=8,
            flush_interval=8,
            jitter_interval=0,
            retry_interval=1000,
        )

        # create influx write lock for thread safety (used only when writing output timeseries to InfluxDB)
        self.is_manager_locked = False

        self.BUCKET_INFLUX_DATA: str = config_handler.read_option(
            "influx2", "bucket_influx_data"
        )
        self.BUCKET_OUT_CML: str = config_handler.read_option(
            "influx2", "bucket_out_cml"
        )

    def check_connection(self) -> bool:
        return self.client.ping()

    def is_aligned_10_min(self, ts: datetime) -> bool:
        return ts.minute % 10 == 0 and ts.second == 0 and ts.microsecond == 0

    # ------------------------------------------------------------------
    # Flux query builder (with pivot)
    # ------------------------------------------------------------------

    def _build_flux_query(
        self,
        start_str: str,
        end_str: str,
        ips_regex: str,
        interval_str: str,
    ) -> str:
        """
        Build a Flux query that:
        - ranges on [start, end)
        - filters by fields of interest
        - filters agent_host by regex (chunked IPs)
        - aggregateWindow (server-side resampling)
        - pivot to wide format: columns = fields, rows = (_time, agent_host)
        """
        flux = f"""
        from(bucket: "{self.BUCKET_INFLUX_DATA}")
        |> range(start: {start_str}, stop: {end_str})
        |> filter(fn: (r) =>
            r["_field"] == "Teplota" or
            r["_field"] == "VysilaciVykon" or
            r["_field"] == "Vysilany_Vykon" or
            r["_field"] == "PrijimanaUroven" or
            r["_field"] == "Signal"
        )
        |> filter(fn: (r) => r["agent_host"] =~ /^({ips_regex})$/)
        |> aggregateWindow(every: {interval_str}, fn: mean, createEmpty: true)
        |> pivot(rowKey:["_time","agent_host"], columnKey: ["_field"], valueColumn: "_value")
        """
        return flux

    # ------------------------------------------------------------------
    # internal low-level query helper: chunked DataFrame fetch
    # ------------------------------------------------------------------

    @measure_time
    def _raw_query_chunks_df(
        self,
        start_str: str,
        end_str: str,
        ip_list: List[str],
        interval_str: str,
        chunk_size: int = 300,
    ) -> pd.DataFrame:
        """
        Chunked query using pandas DataFrame with server-side pivot.

        Returns a single DataFrame with columns:
            _time (datetime64[ns, UTC])
            agent_host (str)
            temperature (float)
            rx_power (float)
            tx_power (float)
        """

        if not ip_list:
            return pd.DataFrame(
                columns=["_time", "agent_host", "temperature", "rx_power", "tx_power"]
            )

        # escape IPs for safe regex
        escaped_ips = [re.escape(ip) for ip in ip_list]

        df_list: list[pd.DataFrame] = []

        for i in range(0, len(escaped_ips), chunk_size):
            ip_chunk = escaped_ips[i : i + chunk_size]
            ips_regex = "|".join(ip_chunk)

            flux = self._build_flux_query(
                start_str=start_str,
                end_str=end_str,
                ips_regex=ips_regex,
                interval_str=interval_str,
            )

            try:
                chunk_df = self.qapi.query_data_frame(flux)
            except Exception as e:
                # In case of "compiled too big" or similar, retry with smaller chunks
                if "compiled too big" in str(e) and chunk_size > 50:
                    logger.warning(
                        "Flux query compiled too big, retrying with smaller chunk_size=%d",
                        chunk_size // 2,
                    )
                    return self._raw_query_chunks_df(
                        start_str=start_str,
                        end_str=end_str,
                        ip_list=ip_list,
                        interval_str=interval_str,
                        chunk_size=chunk_size // 2,
                    )
                else:
                    logger.error(
                        "Error occurred during InfluxDB read query, skipping chunk. Error: %s",
                        e,
                    )
                    continue

            # query_data_frame may return list-of-DFs or a single DF
            if isinstance(chunk_df, list):
                for df_part in chunk_df:
                    if isinstance(df_part, pd.DataFrame) and not df_part.empty:
                        df_list.append(df_part)
            elif isinstance(chunk_df, pd.DataFrame) and not chunk_df.empty:
                df_list.append(chunk_df)

        if not df_list:
            return pd.DataFrame(
                columns=["_time", "agent_host", "temperature", "rx_power", "tx_power"]
            )

        # Concatenate all chunks once
        df = pd.concat(df_list, ignore_index=True)
        if df.empty:
            return df

        # Some Influx versions use a column named "result" or multiindex columns.
        # Flatten columns if needed.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(c) for c in col if c != ""]) for col in df.columns
            ]

        # Ensure we have the necessary columns
        if "_time" not in df.columns or "agent_host" not in df.columns:
            return pd.DataFrame(
                columns=["_time", "agent_host", "temperature", "rx_power", "tx_power"]
            )

        # Normalize time to UTC datetime
        df["_time"] = pd.to_datetime(df["_time"], utc=True)

        # ------------------------------------------------------------------
        # Map raw fields → logical fields: temperature, rx_power, tx_power
        # ------------------------------------------------------------------
        # Temperature
        if "Teplota" in df.columns:
            df["temperature"] = df["Teplota"]
        else:
            df["temperature"] = np.nan

        # Rx power: prefer "PrijimanaUroven", fall back to "Signal"
        rx_series = None
        if "PrijimanaUroven" in df.columns:
            rx_series = df["PrijimanaUroven"]
        if "Signal" in df.columns:
            if rx_series is None:
                rx_series = df["Signal"]
            else:
                rx_series = rx_series.fillna(df["Signal"])
        if rx_series is None:
            df["rx_power"] = np.nan
        else:
            df["rx_power"] = rx_series

        # Tx power: prefer "VysilaciVykon", fall back to "Vysilany_Vykon"
        tx_series = None
        if "VysilaciVykon" in df.columns:
            tx_series = df["VysilaciVykon"]
        if "Vysilany_Vykon" in df.columns:
            if tx_series is None:
                tx_series = df["Vysilany_Vykon"]
            else:
                tx_series = tx_series.fillna(df["Vysilany_Vykon"])
        if tx_series is None:
            df["tx_power"] = np.nan
        else:
            df["tx_power"] = tx_series

        # Keep only relevant columns
        df = df[["_time", "agent_host", "temperature", "rx_power", "tx_power"]]

        # Drop rows where all three logical fields are NaN
        df = df.dropna(subset=["temperature", "rx_power", "tx_power"], how="all")

        # If you suspect duplicates from Influx, you can uncomment this:
        # df = df.sort_values(["agent_host", "_time"]).drop_duplicates(
        #     ["agent_host", "_time"], keep="last"
        # )

        return df

    @measure_time
    def query_units(
        self,
        ips: List[str],
        start: datetime,
        end: datetime,
        interval: int,
        rolling_values: int = None,
        compensate_historic: bool = False,
    ) -> pd.DataFrame:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses (tags in InfluxDB).
        Returns a pandas DataFrame in wide form.

        :param ips: list of IP addresses of CMLs to query
        :param start: datetime with start of the query interval (UTC)
        :param end: datetime with end of the query interval (UTC)
        :param interval: time interval in minutes
        :param rolling_values: number of samples for rolling window (historic compensation)
        :param compensate_historic: whether to compensate for historic rolling window
        :return: pandas DataFrame with columns [_time, agent_host, temperature, rx_power, tx_power]
        """

        if not ips:
            return pd.DataFrame(
                columns=["_time", "agent_host", "temperature", "rx_power", "tx_power"]
            )

        # historic compensation -> extend start backwards
        if compensate_historic and rolling_values is not None:
            num_nan_samples = rolling_values
            compensation_seconds = num_nan_samples * interval * 60
            start = start - timedelta(seconds=compensation_seconds)

        # align start and end to the interval grid
        start = start - timedelta(
            minutes=start.minute % interval,
            seconds=start.second,
            microseconds=start.microsecond,
        )
        end = end - timedelta(
            minutes=end.minute % interval,
            seconds=end.second,
            microseconds=end.microsecond,
        )

        # convert params to query substrings
        start_str = datetime_rfc(start)
        end_str = datetime_rfc(end)
        interval_str = f"{interval * 60}s"  # time in seconds

        return self._raw_query_chunks_df(
            start_str=start_str,
            end_str=end_str,
            ip_list=ips,
            interval_str=interval_str,
            chunk_size=300,
        )

    def query_units_realtime(
        self,
        ips: List[str],
        realtime_window_str: str,
        interval: int,
    ) -> pd.DataFrame:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses.
        Query is done for the time interval defined by 'realtime_window_str'.

        :param ips: list of IP addresses of CMLs to query
        :param realtime_window_str: string describing selected moving time window
        :param interval: time interval in minutes
        :return: pandas DataFrame with columns [_time, agent_host, temperature, rx_power, tx_power]
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
            ips=ips,
            start=start,
            end=end,
            interval=interval,
            rolling_values=None,
            compensate_historic=False,
        )

    def write_points(self, points, bucket):
        with InfluxDBClient.from_config_file(config_handler.config_path) as client_out:
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


# global instance of InfluxManager, accessible from all modules
influx_man = InfluxManager()
