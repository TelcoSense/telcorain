import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from telcorain.database.influx_manager import InfluxManager
from telcorain.handlers import logger
from telcorain.procedures.exceptions import ProcessingException
from telcorain.helpers import measure_time, MwLink


REALTIME_DELTA_MAP: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "3h": timedelta(hours=3),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "2d": timedelta(days=2),
    "7d": timedelta(days=7),
    "14d": timedelta(days=14),
    "30d": timedelta(days=30),
}

REALTIME_BUFFER_OVERLAP_STEPS = 2


@dataclass
class RealtimeInfluxBuffer:
    """
    Stateful raw Influx frame cache for realtime mode.

    The cache is row-based, so it naturally supports links appearing/disappearing
    between fetches as long as the IP selection itself stays the same.
    """

    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ips: frozenset[str] = field(default_factory=frozenset)
    interval_minutes: Optional[int] = None
    include_temperature: Optional[bool] = None
    last_window_end: Optional[datetime] = None

    def is_compatible(
        self,
        ips: List[str],
        interval_minutes: int,
        include_temperature: bool,
    ) -> bool:
        return (
            self.ips == frozenset(ips)
            and self.interval_minutes == interval_minutes
            and self.include_temperature == include_temperature
        )


def _make_base_df(include_temperature: bool) -> pd.DataFrame:
    base_cols = ["_time", "agent_host", "rx_power", "tx_power"]
    if include_temperature:
        base_cols.insert(2, "temperature")
    return pd.DataFrame(columns=base_cols)


def _trim_realtime_buffer_df(
    df: Optional[pd.DataFrame],
    *,
    include_temperature: bool,
    ips: List[str],
    window_start: datetime,
) -> pd.DataFrame:
    base_cols = _make_base_df(include_temperature).columns.tolist()
    if df is None or df.empty:
        return pd.DataFrame(columns=base_cols)

    trimmed = df.copy()
    trimmed["_time"] = pd.to_datetime(trimmed["_time"], utc=True)
    trimmed = trimmed[trimmed["_time"] >= pd.Timestamp(window_start)]
    trimmed = trimmed[trimmed["agent_host"].isin(ips)]

    if trimmed.empty:
        return pd.DataFrame(columns=base_cols)

    trimmed = trimmed.sort_values(["agent_host", "_time"])
    trimmed = trimmed.drop_duplicates(subset=["_time", "agent_host"], keep="last")

    for col in base_cols:
        if col not in trimmed.columns:
            trimmed[col] = np.nan

    return trimmed[base_cols].reset_index(drop=True)


def _load_realtime_data_from_influxdb(
    influx_man: InfluxManager,
    *,
    ips: List[str],
    interval_minutes: int,
    realtime_timewindow: str,
    include_temperature: bool,
    realtime_buffer: Optional[RealtimeInfluxBuffer],
    force_refresh: bool,
) -> tuple[pd.DataFrame, RealtimeInfluxBuffer]:
    if realtime_timewindow not in REALTIME_DELTA_MAP:
        raise ValueError(f"Unsupported realtime window: {realtime_timewindow}")

    end = datetime.now(timezone.utc)
    window_start = end - REALTIME_DELTA_MAP[realtime_timewindow]

    buffer = realtime_buffer or RealtimeInfluxBuffer()
    buffer_is_compatible = buffer.is_compatible(
        ips=ips,
        interval_minutes=interval_minutes,
        include_temperature=include_temperature,
    )
    needs_full_refresh = (
        force_refresh or not buffer_is_compatible or buffer.last_window_end is None
    )

    if needs_full_refresh:
        logger.debug(
            "Realtime Influx fetch: full refresh (force=%s, compatible=%s, has_cursor=%s).",
            force_refresh,
            buffer_is_compatible,
            buffer.last_window_end is not None,
        )
        df = influx_man.query_units(
            ips=ips,
            start=window_start,
            end=end,
            interval=interval_minutes,
            rolling_values=None,
            compensate_historic=False,
            include_temperature=include_temperature,
        )
    else:
        overlap = timedelta(
            minutes=max(interval_minutes * REALTIME_BUFFER_OVERLAP_STEPS, interval_minutes)
        )
        fetch_anchor = (
            pd.to_datetime(buffer.df["_time"].max(), utc=True).to_pydatetime()
            if not buffer.df.empty
            else buffer.last_window_end
        )
        fetch_start = max(window_start, fetch_anchor - overlap)

        logger.debug(
            "Realtime Influx fetch: incremental refresh from %s to %s (cached rows=%d).",
            fetch_start,
            end,
            0 if buffer.df is None else len(buffer.df),
        )

        if fetch_start >= end:
            df = buffer.df
        else:
            tail_df = influx_man.query_units(
                ips=ips,
                start=fetch_start,
                end=end,
                interval=interval_minutes,
                rolling_values=None,
                compensate_historic=False,
                include_temperature=include_temperature,
            )
            df = (
                pd.concat([buffer.df, tail_df], ignore_index=True)
                if buffer.df is not None and not buffer.df.empty
                else tail_df
            )

    trimmed = _trim_realtime_buffer_df(
        df,
        include_temperature=include_temperature,
        ips=ips,
        window_start=window_start,
    )
    updated_buffer = RealtimeInfluxBuffer(
        df=trimmed,
        ips=frozenset(ips),
        interval_minutes=interval_minutes,
        include_temperature=include_temperature,
        last_window_end=end,
    )
    return trimmed, updated_buffer


def get_ips_from_links_dict(
    selected_links: Dict[int, int],
    links: Dict[int, MwLink],
) -> List[str]:
    """
    Build a list of IP addresses from the selected links.

    selected_links: dict[link_id] -> any truthy value = enabled, falsy = disabled.
    links:          dict[link_id] -> MwLink
    """
    if not selected_links:
        raise ValueError("Empty selection array.")

    ips: set[str] = set()
    for link_id, enabled in selected_links.items():
        if not enabled:
            continue
        link = links.get(link_id)
        if link is None:
            continue
        ips.add(link.ip_a)
        ips.add(link.ip_b)

    return list(ips)


@measure_time
def load_data_from_influxdb(
    influx_man: InfluxManager,
    config: dict,
    selected_links: Dict[int, int],
    links: Dict[int, MwLink],
    log_run_id: str = "default",
    realtime: bool = False,
    realtime_timewindow: str = "1d",
    realtime_buffer: Optional[RealtimeInfluxBuffer] = None,
    force_realtime_refresh: bool = False,
) -> Tuple[pd.DataFrame, List[int], List[str], Optional[RealtimeInfluxBuffer]]:
    """
    Fetch data from InfluxDB and return:
      - df: wide dataframe with columns [_time, agent_host, rx_power, tx_power] + optional [temperature]
      - missing_links: list of link_id that are missing (based on missing IPs)
      - ips: queried IP list
    """
    try:
        ips = get_ips_from_links_dict(selected_links, links)

        # temperature is needed only when used later
        wd_cfg = config.get("wet_dry", {})
        need_temperature = bool(
            wd_cfg.get("is_temp_filtered", False)
            or wd_cfg.get("is_temp_compensated", False)
        )

        if realtime:
            df, realtime_buffer = _load_realtime_data_from_influxdb(
                influx_man,
                ips=ips,
                interval_minutes=config["time"]["step"],
                realtime_timewindow=realtime_timewindow,
                include_temperature=need_temperature,
                realtime_buffer=realtime_buffer,
                force_refresh=force_realtime_refresh,
            )
        else:
            # compute warm-up samples for historic compensation
            hist_cfg = config.get("historic", {})
            compensate = bool(hist_cfg.get("compensate_historic", False))

            warmup_samples = None
            if compensate:
                rolling_vals = int(wd_cfg.get("rolling_values", 0) or 0)
                baseline_samples = int(wd_cfg.get("baseline_samples", 0) or 0)

                warmup_samples = max(rolling_vals, baseline_samples)

                # If CNN is used, also respect its internal warm-up
                if wd_cfg.get("is_mlp_enabled", False):
                    try:
                        from telcorain.procedures.wet_dry.cnn import (
                            CNN_OUTPUT_LEFT_NANS_LENGTH,
                        )

                        warmup_samples = max(
                            warmup_samples, int(CNN_OUTPUT_LEFT_NANS_LENGTH)
                        )
                    except Exception:
                        pass

                # Add extra warmup for 1-hour rolling sum (hour_sum feature)
                hs_cfg = config.get("hour_sum", {})
                if bool(hs_cfg.get("enabled", False)):
                    ts = int(config["time"]["output_step"])  # minutes, e.g. 10
                    win_min = int(hs_cfg.get("window_minutes", 60))
                    if ts > 0 and win_min > 0:
                        win_steps = int(round(win_min / ts))
                        hour_sum_warmup = max(0, win_steps - 1)
                        warmup_samples = max(int(warmup_samples or 0), hour_sum_warmup)

                if warmup_samples <= 0:
                    warmup_samples = None

            df = influx_man.query_units(
                ips=ips,
                start=config["time"]["start"],
                end=config["time"]["end"],
                interval=config["time"]["step"],
                rolling_values=warmup_samples,
                compensate_historic=compensate,
                include_temperature=need_temperature,
            )

        if df is None or df.empty:
            logger.info("[%s] Influx returned empty DataFrame.", log_run_id)
            base_cols = ["_time", "agent_host", "rx_power", "tx_power"]
            if need_temperature:
                base_cols.insert(2, "temperature")
            empty_df = pd.DataFrame(columns=base_cols)
            return empty_df, list(links.keys()), ips, realtime_buffer

        # Determine missing IPs based on DataFrame
        present_ips = set(df["agent_host"].unique())
        missing_links: List[int] = []

        for ip in ips:
            if ip not in present_ips:
                for link_id, link in links.items():
                    if link.ip_a == ip or link.ip_b == ip:
                        missing_links.append(link_id)
                        break

        logger.info(
            "[%s] Querying done. Got data for %d IPs (of %d selected IPs).",
            log_run_id,
            len(present_ips),
            len(ips),
        )

        return df, missing_links, ips, realtime_buffer

    except BaseException as error:
        logger.error(
            "[%s] An unexpected error occurred during InfluxDB query: %s %s.\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
        )
        traceback.print_exc()
        raise ProcessingException("Error occurred during InfluxDB query.")


@measure_time
def convert_to_link_datasets(
    selected_links: Dict[int, int],
    links: Dict[int, MwLink],
    df: pd.DataFrame,
    missing_links: List[int],
    log_run_id: str = "default",
) -> List[xr.Dataset]:
    if df is None or df.empty:
        logger.warning("[%s] Empty DF in convert_to_link_datasets.", log_run_id)
        return []

    # ------------------------------------------------------------------
    # Global sort + deduplicate per (agent_host, _time)
    # ------------------------------------------------------------------
    df = df.sort_values(["agent_host", "_time"])
    df = df.drop_duplicates(subset=["agent_host", "_time"], keep="last")
    df = df.set_index("_time")

    groups: Dict[str, pd.DataFrame] = dict(tuple(df.groupby("agent_host", sort=False)))

    calc_data: List[xr.Dataset] = []

    def build_channel_fast(
        link_obj: MwLink,
        df_rx: pd.DataFrame,
        df_tx: Optional[pd.DataFrame],
        channel_id: str,
        freq_tx: int,
    ) -> xr.Dataset:
        """Optimized channel builder with robust index handling."""
        df_rx = df_rx.sort_index()
        if df_rx.index.has_duplicates:
            df_rx = df_rx[~df_rx.index.duplicated(keep="last")]

        times = df_rx.index.values

        rsl = df_rx["rx_power"].to_numpy(dtype=float)

        if "temperature" in df_rx.columns:
            temperature_rx = df_rx["temperature"].fillna(0.0).to_numpy(dtype=float)
        else:
            temperature_rx = np.zeros_like(rsl, dtype=float)

        if df_tx is None or df_tx.empty:
            tsl = np.zeros_like(rsl)
            temperature_tx = np.zeros_like(rsl, dtype=float)
        else:
            df_tx = df_tx.sort_index()
            if df_tx.index.has_duplicates:
                df_tx = df_tx[~df_tx.index.duplicated(keep="last")]

            aligned_tx = df_tx.reindex(df_rx.index)

            tsl = aligned_tx["tx_power"].fillna(0.0).to_numpy(dtype=float)

            if "temperature" in aligned_tx.columns:
                temperature_tx = (
                    aligned_tx["temperature"].fillna(0.0).to_numpy(dtype=float)
                )
            else:
                temperature_tx = np.zeros_like(tsl, dtype=float)

        if link_obj.tech in ["summit", "summit_bt"]:
            rsl = -rsl

        ds = xr.Dataset(
            data_vars=dict(
                tsl=("time", tsl),
                rsl=("time", rsl),
                temperature_rx=("time", temperature_rx),
                temperature_tx=("time", temperature_tx),
            ),
            coords=dict(
                time=times.astype("datetime64[ns]"),
                channel_id=channel_id,
                cml_id=link_obj.link_id,
                site_a_latitude=link_obj.latitude_a,
                site_b_latitude=link_obj.latitude_b,
                site_a_longitude=link_obj.longitude_a,
                site_b_longitude=link_obj.longitude_b,
                frequency=freq_tx / 1000.0,
                polarization=link_obj.polarization,
                length=link_obj.distance,
            ),
        )
        return ds

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for link_id, enabled in selected_links.items():
        if not enabled:
            continue

        link = links.get(link_id)
        if link is None:
            continue

        ip_a, ip_b = link.ip_a, link.ip_b

        if ip_a not in groups or ip_b not in groups:
            continue

        df_a = groups[ip_a]
        df_b = groups[ip_b]

        # avoid pycomlink crash
        if link.freq_a == link.freq_b:
            link.freq_a += 1

        ch_ab = build_channel_fast(
            link_obj=link,
            df_rx=df_a,
            df_tx=df_b,
            channel_id="A(rx)_B(tx)",
            freq_tx=link.freq_b,
        )

        ch_ba = build_channel_fast(
            link_obj=link,
            df_rx=df_b,
            df_tx=df_a,
            channel_id="B(rx)_A(tx)",
            freq_tx=link.freq_a,
        )

        calc_data.append(xr.concat([ch_ab, ch_ba], dim="channel_id"))

    return calc_data
