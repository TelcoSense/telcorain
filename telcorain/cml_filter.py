from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from telcorain.handlers import logger
from telcorain.helpers import MwLink

if TYPE_CHECKING:
    from telcorain.database.influx_manager import InfluxManager


REALTIME_WINDOWS: dict[str, timedelta] = {
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


@dataclass
class CmlQualityResult:
    inspected_count: int
    accepted_ids: set[int] = field(default_factory=set)
    rejected_reasons: dict[int, str] = field(default_factory=dict)

    @property
    def rejected_ids(self) -> set[int]:
        return set(self.rejected_reasons)


def is_cml_quality_filter_enabled(config: dict) -> bool:
    return _as_bool(config.get("cml_filter", {}).get("enabled", False))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class CmlQualityFilter:
    """
    Stateful CML quality filter.

    The filter stores rejected CML IDs in memory and applies them to the active
    selection. Metadata stays available, so later scans can re-enable recovered
    links without restarting the process.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rejected_cml_ids: set[int] = set()
        self.last_result: Optional[CmlQualityResult] = None
        self.last_run_id: Optional[int] = None

    def enabled(self) -> bool:
        return is_cml_quality_filter_enabled(self.config)

    def should_run(self, *, is_historic: bool, realtime_run: int) -> bool:
        if not self.enabled():
            return False
        cfg = self.config.get("cml_filter", {})
        if is_historic:
            return True
        if _as_bool(cfg.get("run_realtime_at_start", True)) and realtime_run == 1:
            return True
        interval = int(cfg.get("realtime_interval_runs", 1000) or 0)
        return interval > 0 and realtime_run > 0 and realtime_run % interval == 0

    def apply_known_exclusions(self, selection: dict[int, int | bool]) -> dict[int, bool]:
        if not self.rejected_cml_ids:
            return {int(k): bool(v) for k, v in selection.items()}
        return {
            int(link_id): bool(enabled) and int(link_id) not in self.rejected_cml_ids
            for link_id, enabled in selection.items()
        }

    def inspect_and_filter(
        self,
        *,
        influx_man: InfluxManager,
        links: dict[int, MwLink],
        selection: dict[int, int | bool],
        log_run_id: str,
        is_historic: bool,
        realtime_timewindow: str,
        realtime_run: int = 0,
    ) -> dict[int, bool]:
        if not self.enabled():
            return {int(k): bool(v) for k, v in selection.items()}

        result = inspect_cml_quality(
            influx_man=influx_man,
            config=self.config,
            links=links,
            log_run_id=log_run_id,
            is_historic=is_historic,
            realtime_timewindow=realtime_timewindow,
        )

        self.last_result = result
        self.last_run_id = realtime_run if not is_historic else None
        self.rejected_cml_ids = result.rejected_ids

        filtered_selection = {
            int(link_id): int(link_id) in result.accepted_ids
            for link_id in links
        }

        logger.info(
            "[%s] CML quality filter: accepted %d/%d links, rejected %d.",
            log_run_id,
            len(result.accepted_ids),
            result.inspected_count,
            len(result.rejected_ids),
        )
        if result.rejected_reasons:
            preview = ", ".join(
                f"{link_id}:{reason}"
                for link_id, reason in list(sorted(result.rejected_reasons.items()))[:10]
            )
            if len(result.rejected_reasons) > 10:
                preview += ", ..."
            logger.info("[%s] CML quality filter rejected IDs: %s", log_run_id, preview)

        return filtered_selection


def inspect_cml_quality(
    *,
    influx_man: InfluxManager,
    config: dict,
    links: dict[int, MwLink],
    log_run_id: str,
    is_historic: bool,
    realtime_timewindow: str,
) -> CmlQualityResult:
    selected_ids = [int(link_id) for link_id in links]
    result = CmlQualityResult(inspected_count=len(selected_ids))
    if not selected_ids:
        return result

    cfg = config.get("cml_filter", {})
    interval_minutes = int(config["time"]["step"])
    start, end = _inspection_window(config, cfg, is_historic, realtime_timewindow)
    expected_samples = _expected_samples(start, end, interval_minutes)

    ips = _ips_for_selected_links(links, selected_ids)
    df = influx_man.query_units(
        ips=ips,
        start=start,
        end=end,
        interval=interval_minutes,
        rolling_values=None,
        compensate_historic=False,
        include_temperature=False,
    )
    if df is None or df.empty:
        for link_id in selected_ids:
            result.rejected_reasons[link_id] = "no_influx_data"
        return result

    df = df.copy()
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    groups = {
        str(ip): group.sort_values("_time").drop_duplicates(["_time"], keep="last")
        for ip, group in df.groupby("agent_host", sort=False)
    }

    for link_id in selected_ids:
        link = links[link_id]
        thresholds = _thresholds_for_link(config, link)
        reason = _reject_reason_for_link(
            link=link,
            groups=groups,
            expected_samples=expected_samples,
            thresholds=thresholds,
        )
        if reason is None:
            result.accepted_ids.add(link_id)
        else:
            result.rejected_reasons[link_id] = reason

    return result


def _inspection_window(
    config: dict,
    filter_cfg: dict,
    is_historic: bool,
    realtime_timewindow: str,
) -> tuple[datetime, datetime]:
    if is_historic:
        return config["time"]["start"], config["time"]["end"]

    window_name = str(filter_cfg.get("realtime_quality_window", "") or realtime_timewindow)
    delta = REALTIME_WINDOWS.get(window_name)
    if delta is None:
        logger.warning(
            "Unsupported cml_filter.realtime_quality_window=%s; falling back to %s.",
            window_name,
            realtime_timewindow,
        )
        delta = REALTIME_WINDOWS.get(realtime_timewindow, timedelta(days=1))

    end = datetime.now(timezone.utc)
    return end - delta, end


def _expected_samples(start: datetime, end: datetime, interval_minutes: int) -> int:
    seconds = max(0.0, (end - start).total_seconds())
    interval_seconds = max(1, interval_minutes * 60)
    return max(1, int(np.ceil(seconds / interval_seconds)))


def _ips_for_selected_links(links: dict[int, MwLink], selected_ids: list[int]) -> list[str]:
    ips: set[str] = set()
    for link_id in selected_ids:
        link = links[link_id]
        ips.add(str(link.ip_a))
        ips.add(str(link.ip_b))
    return sorted(ips)


def _thresholds(config: dict) -> dict[str, float | int | bool]:
    cfg = config.get("cml_filter", {})
    return {
        "min_endpoint_coverage": float(cfg.get("min_endpoint_coverage", 0.5)),
        "min_channel_coverage": float(cfg.get("min_channel_coverage", 0.5)),
        "min_valid_samples": int(cfg.get("min_valid_samples", 12)),
        "minimum_good_channels": int(cfg.get("minimum_good_channels", 1)),
        "rsl_min_dbm": float(cfg.get("rsl_min_dbm", -120.0)),
        "rsl_max_dbm": float(cfg.get("rsl_max_dbm", 120.0)),
        "effective_rsl_min_dbm": float(cfg.get("effective_rsl_min_dbm", -70.0)),
        "tsl_min_dbm": float(cfg.get("tsl_min_dbm", -20.0)),
        "tsl_max_dbm": float(cfg.get("tsl_max_dbm", 40.0)),
        "trsl_min_db": float(cfg.get("trsl_min_db", 0.0)),
        "trsl_max_db": float(cfg.get("trsl_max_db", 99.0)),
        "min_trsl_p05_p95_range_db": float(cfg.get("min_trsl_p05_p95_range_db", 0.02)),
        "require_tsl": _as_bool(cfg.get("require_tsl", False)),
    }


def _thresholds_for_link(config: dict, link: MwLink) -> dict[str, float | int | bool]:
    thresholds = _thresholds(config)
    tech = str(link.tech).strip()
    if not tech:
        return thresholds

    for section_name in (
        f"cml_filter:{tech}",
        f"cml_filter.technology.{tech}",
        f"technology:{tech}",
    ):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in list(thresholds):
            if key not in section:
                continue
            if key == "require_tsl":
                thresholds[key] = _as_bool(section[key])
            elif isinstance(thresholds[key], int) and not isinstance(thresholds[key], bool):
                thresholds[key] = int(section[key])
            elif isinstance(thresholds[key], bool):
                thresholds[key] = _as_bool(section[key])
            else:
                thresholds[key] = float(section[key])
        break
    return thresholds


def _reject_reason_for_link(
    *,
    link: MwLink,
    groups: dict[str, pd.DataFrame],
    expected_samples: int,
    thresholds: dict[str, float | int | bool],
) -> Optional[str]:
    df_a = groups.get(str(link.ip_a))
    df_b = groups.get(str(link.ip_b))
    if df_a is None or df_a.empty:
        return "missing_ip_a"
    if df_b is None or df_b.empty:
        return "missing_ip_b"

    endpoint_a = _endpoint_has_enough_rx(df_a, expected_samples, thresholds)
    endpoint_b = _endpoint_has_enough_rx(df_b, expected_samples, thresholds)
    if not endpoint_a:
        return "low_rx_coverage_ip_a"
    if not endpoint_b:
        return "low_rx_coverage_ip_b"

    good_channels = 0
    if _channel_is_usable(df_rx=df_a, df_tx=df_b, link=link, expected_samples=expected_samples, thresholds=thresholds):
        good_channels += 1
    if _channel_is_usable(df_rx=df_b, df_tx=df_a, link=link, expected_samples=expected_samples, thresholds=thresholds):
        good_channels += 1

    minimum_good_channels = int(thresholds["minimum_good_channels"])
    if good_channels < minimum_good_channels:
        return "bad_trsl_quality"
    return None


def _endpoint_has_enough_rx(
    df: pd.DataFrame,
    expected_samples: int,
    thresholds: dict[str, float | int | bool],
) -> bool:
    rx = pd.to_numeric(df["rx_power"], errors="coerce")
    valid = (
        np.isfinite(rx)
        & (rx != 0.0)
        & (rx >= float(thresholds["rsl_min_dbm"]))
        & (rx <= float(thresholds["rsl_max_dbm"]))
    )
    valid_count = int(valid.sum())
    return (
        valid_count >= int(thresholds["min_valid_samples"])
        and valid_count / expected_samples >= float(thresholds["min_endpoint_coverage"])
    )


def _channel_is_usable(
    *,
    df_rx: pd.DataFrame,
    df_tx: pd.DataFrame,
    link: MwLink,
    expected_samples: int,
    thresholds: dict[str, float | int | bool],
) -> bool:
    rx = df_rx[["_time", "rx_power"]].rename(columns={"rx_power": "rx_power_rx"})
    tx = df_tx[["_time", "tx_power"]].rename(columns={"tx_power": "tx_power_tx"})
    merged = rx.merge(tx, on="_time", how="left")
    if merged.empty:
        return False

    rsl = pd.to_numeric(merged["rx_power_rx"], errors="coerce").to_numpy(dtype=float)
    if link.tech in {"summit", "summit_bt"}:
        rsl = -rsl

    tx_raw = pd.to_numeric(merged["tx_power_tx"], errors="coerce").to_numpy(dtype=float)
    tx_present = np.isfinite(tx_raw)
    tsl = np.where(tx_present, tx_raw, 0.0)
    tx_ok = (
        tx_present
        & (tx_raw >= float(thresholds["tsl_min_dbm"]))
        & (tx_raw < float(thresholds["tsl_max_dbm"]))
    )
    if not bool(thresholds["require_tsl"]):
        tx_ok = tx_ok | ~tx_present

    trsl = tsl - rsl
    valid = (
        np.isfinite(rsl)
        & (rsl != 0.0)
        & (rsl > float(thresholds["effective_rsl_min_dbm"]))
        & tx_ok
        & np.isfinite(trsl)
        & (trsl >= float(thresholds["trsl_min_db"]))
        & (trsl <= float(thresholds["trsl_max_db"]))
    )
    valid_count = int(np.sum(valid))
    if valid_count < int(thresholds["min_valid_samples"]):
        return False
    if valid_count / expected_samples < float(thresholds["min_channel_coverage"]):
        return False

    valid_trsl = trsl[valid]
    if valid_trsl.size < 2:
        return False
    dynamic_range = float(np.nanpercentile(valid_trsl, 95) - np.nanpercentile(valid_trsl, 5))
    return dynamic_range >= float(thresholds["min_trsl_p05_p95_range_db"])
