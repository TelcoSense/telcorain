import traceback
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from telcorain.database.influx_manager import InfluxManager
from telcorain.handlers import logger
from telcorain.procedures.exceptions import ProcessingException
from telcorain.helpers import measure_time, MwLink


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
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    ...
    try:
        ips = get_ips_from_links_dict(selected_links, links)
        if realtime:
            df = influx_man.query_units_realtime(
                ips=ips,
                realtime_window_str=realtime_timewindow,
                interval=config["time"]["step"],
            )
        else:
            # compute warm-up samples for historic compensation
            hist_cfg = config.get("historic", {})
            compensate = bool(hist_cfg.get("compensate_historic", False))

            warmup_samples = None
            if compensate:
                wd_cfg = config.get("wet_dry", {})
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
                        # Fallback if CNN constant cannot be imported
                        pass

                # ------------------------------------------------------------
                # Add extra warmup for 1-hour rolling sum (hour_sum feature)
                # ------------------------------------------------------------
                hs_cfg = config.get("hour_sum", {})
                if bool(hs_cfg.get("enabled", False)):
                    ts = int(config["time"]["output_step"])  # minutes, e.g. 10
                    win_min = int(hs_cfg.get("window_minutes", 60))
                    if ts > 0 and win_min > 0:
                        win_steps = int(round(win_min / ts))
                        # need (win_steps-1) frames before first output to have full sum
                        hour_sum_warmup = max(0, win_steps - 1)
                        warmup_samples = max(int(warmup_samples or 0), hour_sum_warmup)

                if warmup_samples <= 0:
                    warmup_samples = None

            # Pass warmup_samples to InfluxManager as generic "rolling_values"
            df = influx_man.query_units(
                ips=ips,
                start=config["time"]["start"],
                end=config["time"]["end"],
                interval=config["time"]["step"],
                rolling_values=warmup_samples,
                compensate_historic=compensate,
            )

        if df is None or df.empty:
            logger.info("[%s] Influx returned empty DataFrame.", log_run_id)
            empty_df = pd.DataFrame(
                columns=["_time", "agent_host", "temperature", "rx_power", "tx_power"]
            )
            # treat all links as missing
            return empty_df, list(links.keys()), ips

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

        return df, missing_links, ips

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

    # Pre-group once (fast after sorting + indexing)
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

        # Ensure sorted unique index on RX
        df_rx = df_rx.sort_index()
        if df_rx.index.has_duplicates:
            df_rx = df_rx[~df_rx.index.duplicated(keep="last")]

        times = df_rx.index.values  # sorted, unique

        rsl = df_rx["rx_power"].to_numpy(dtype=float)
        temperature_rx = df_rx["temperature"].fillna(0.0).to_numpy(dtype=float)

        if df_tx is None or df_tx.empty:
            tsl = np.zeros_like(rsl)
            temperature_tx = np.zeros_like(rsl, dtype=float)
        else:
            df_tx = df_tx.sort_index()
            if df_tx.index.has_duplicates:
                df_tx = df_tx[~df_tx.index.duplicated(keep="last")]

            # Align df_tx to df_rx index
            aligned_tx = df_tx.reindex(df_rx.index)

            tsl = aligned_tx["tx_power"].fillna(0.0).to_numpy(dtype=float)
            temperature_tx = aligned_tx["temperature"].fillna(0.0).to_numpy(dtype=float)

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

        # Build channels like in pycomlink
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
