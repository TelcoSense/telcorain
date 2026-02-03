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
        temperature (float)            [optional if include_temperature=False]
        rx_power (float)
        tx_power (float)

      where temperature / rx_power / tx_power are already mapped from:
        Teplota, PrijimanaUroven / Signal, VysilaciVykon / Vysilany_Vykon.
    """

    def __init__(self):
        super(InfluxManager, self).__init__()

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
        *,
        include_temperature: bool = True,
    ) -> str:
        """
        Build a Flux query that:
        - ranges on [start, end)
        - filters by fields of interest
        - filters agent_host by regex (chunked IPs)
        - aggregateWindow (server-side resampling)
        - pivot to wide format: columns = fields, rows = (_time, agent_host)
        """
        field_filters = [
            'r["_field"] == "VysilaciVykon"',
            'r["_field"] == "Vysilany_Vykon"',
            'r["_field"] == "PrijimanaUroven"',
            'r["_field"] == "Signal"',
        ]
        if include_temperature:
            field_filters.insert(0, 'r["_field"] == "Teplota"')

        fields_expr = " or\n            ".join(field_filters)

        flux = f"""
        from(bucket: "{self.BUCKET_INFLUX_DATA}")
        |> range(start: {start_str}, stop: {end_str})
        |> filter(fn: (r) =>
            {fields_expr}
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
        *,
        include_temperature: bool = True,
        chunk_size: int = 300,
    ) -> pd.DataFrame:
        """
        Chunked query using pandas DataFrame with server-side pivot.

        Returns a single DataFrame with columns:
            _time (datetime64[ns, UTC])
            agent_host (str)
            rx_power (float)
            tx_power (float)
            temperature (float)  [only if include_temperature=True]
        """
        base_cols = ["_time", "agent_host", "rx_power", "tx_power"]
        if include_temperature:
            base_cols.insert(2, "temperature")

        if not ip_list:
            return pd.DataFrame(columns=base_cols)

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
                include_temperature=include_temperature,
            )

            try:
                chunk_df = self.qapi.query_data_frame(flux)
            except Exception as e:
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
                        include_temperature=include_temperature,
                        chunk_size=chunk_size // 2,
                    )
                logger.error(
                    "Error occurred during InfluxDB read query, skipping chunk. Error: %s",
                    e,
                )
                continue

            if isinstance(chunk_df, list):
                for df_part in chunk_df:
                    if isinstance(df_part, pd.DataFrame) and not df_part.empty:
                        df_list.append(df_part)
            elif isinstance(chunk_df, pd.DataFrame) and not chunk_df.empty:
                df_list.append(chunk_df)

        if not df_list:
            return pd.DataFrame(columns=base_cols)

        df = pd.concat(df_list, ignore_index=True)
        if df.empty:
            return pd.DataFrame(columns=base_cols)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(c) for c in col if c != ""]) for col in df.columns
            ]

        if "_time" not in df.columns or "agent_host" not in df.columns:
            return pd.DataFrame(columns=base_cols)

        df["_time"] = pd.to_datetime(df["_time"], utc=True)

        # Temperature
        if include_temperature:
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
        df["rx_power"] = np.nan if rx_series is None else rx_series

        # Tx power: prefer "VysilaciVykon", fall back to "Vysilany_Vykon"
        tx_series = None
        if "VysilaciVykon" in df.columns:
            tx_series = df["VysilaciVykon"]
        if "Vysilany_Vykon" in df.columns:
            if tx_series is None:
                tx_series = df["Vysilany_Vykon"]
            else:
                tx_series = tx_series.fillna(df["Vysilany_Vykon"])
        df["tx_power"] = np.nan if tx_series is None else tx_series

        # Keep only relevant columns
        df = df[base_cols]

        # Drop rows where all logical fields are NaN
        drop_cols = ["rx_power", "tx_power"]
        if include_temperature:
            drop_cols = ["temperature"] + drop_cols
        df = df.dropna(subset=drop_cols, how="all")

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
        *,
        include_temperature: bool = True,
    ) -> pd.DataFrame:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses (tags in InfluxDB).
        Returns a pandas DataFrame in wide form.
        """
        base_cols = ["_time", "agent_host", "rx_power", "tx_power"]
        if include_temperature:
            base_cols.insert(2, "temperature")

        if not ips:
            return pd.DataFrame(columns=base_cols)

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

        start_str = datetime_rfc(start)
        end_str = datetime_rfc(end)
        interval_str = f"{interval * 60}s"

        return self._raw_query_chunks_df(
            start_str=start_str,
            end_str=end_str,
            ip_list=ips,
            interval_str=interval_str,
            include_temperature=include_temperature,
            chunk_size=300,
        )

    def query_units_realtime(
        self,
        ips: List[str],
        realtime_window_str: str,
        interval: int,
        *,
        include_temperature: bool = True,
    ) -> pd.DataFrame:
        """
        Query InfluxDB for CMLs defined in 'ips' as list of their IP addresses.
        Query is done for the time interval defined by 'realtime_window_str'.
        """
        delta_map = {
            "1h": timedelta(hours=1),
            "3h": timedelta(hours=3),
            "4h": timedelta(hours=4),
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
            include_temperature=include_temperature,
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
