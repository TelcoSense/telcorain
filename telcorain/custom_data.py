from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from telcorain.dataprocessing import convert_to_link_datasets, load_data_from_influxdb
from telcorain.handlers import logger
from telcorain.helpers import MwLink, calc_distance, RAIN_COLORS, RAIN_THRESH
from telcorain.procedures.exceptions import ProcessingException


@dataclass
class SourceDataBundle:
    calc_data: list[xr.Dataset]
    ips: list[str]
    links: dict[int, MwLink]
    selection: dict[int, bool]
    realtime_buffer: Any = None
    time_min: Optional[pd.Timestamp] = None
    time_max: Optional[pd.Timestamp] = None


def get_data_source_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("data_source", {}).get("mode", "influx")).strip().lower()
    return mode or "influx"


def is_influx_source(config: dict[str, Any]) -> bool:
    return get_data_source_mode(config) == "influx"


def load_links_for_source(
    config: dict[str, Any],
    sql_man,
    *,
    ids=None,
    min_length: float = 0.01,
    max_length: float = float("inf"),
    exclude_ids: bool = True,
) -> dict[int, MwLink]:
    if is_influx_source(config):
        return sql_man.load_metadata(
            ids=ids,
            min_length=min_length,
            max_length=max_length,
            exclude_ids=exclude_ids,
        )
    return {}


def load_calc_data_for_source(
    *,
    influx_man,
    config: dict[str, Any],
    selected_links: dict[int, int | bool],
    links: dict[int, MwLink],
    log_run_id: str = "default",
    realtime: bool = False,
    realtime_timewindow: str = "1d",
    realtime_buffer=None,
    force_realtime_refresh: bool = False,
) -> SourceDataBundle:
    mode = get_data_source_mode(config)
    if mode == "influx":
        df, missing_links, ips, updated_buffer = load_data_from_influxdb(
            influx_man=influx_man,
            config=config,
            selected_links=selected_links,
            links=links,
            log_run_id=log_run_id,
            realtime=realtime,
            realtime_timewindow=realtime_timewindow,
            realtime_buffer=realtime_buffer,
            force_realtime_refresh=force_realtime_refresh,
        )
        calc_data = convert_to_link_datasets(
            selected_links=selected_links,
            links=links,
            df=df,
            missing_links=missing_links,
            log_run_id=log_run_id,
        )
        return SourceDataBundle(
            calc_data=calc_data,
            ips=ips,
            links=links,
            selection={int(k): bool(v) for k, v in selected_links.items()},
            realtime_buffer=updated_buffer,
            time_min=_df_time_min(df),
            time_max=_df_time_max(df),
        )

    if realtime:
        raise ProcessingException(
            "Custom data sources are supported only for one-shot historic/custom runs."
        )

    if mode == "netherlands_raw_csv":
        custom_links, calc_data, time_min, time_max = _load_netherlands_raw_calc_data(
            config=config,
            log_run_id=log_run_id,
        )
        selection = {link_id: True for link_id in custom_links}
        _apply_custom_region_overrides(config, custom_links, log_run_id)
        return SourceDataBundle(
            calc_data=calc_data,
            ips=_ips_from_links(custom_links),
            links=custom_links,
            selection=selection,
            realtime_buffer=None,
            time_min=time_min,
            time_max=time_max,
        )

    if mode not in {"pycomlink_example", "pycomlink_netcdf", "netherlands_raw_csv", "openrainer_tar"}:
        raise ProcessingException(
            f"Unsupported [data_source] mode: {mode!r}. Supported custom modes: pycomlink_example, pycomlink_netcdf, netherlands_raw_csv, openrainer_tar."
        )

    if mode == "openrainer_tar":
        ds = _load_openrainer_cml_dataset(config=config, log_run_id=log_run_id)
    else:
        ds = _load_pycomlink_dataset(config, mode)
    custom_links, source_lookup = _build_links_from_pycomlink_dataset(
        ds,
        min_length=float(config["cml"]["min_length"]),
        max_length=float(config["cml"]["max_length"]),
    )
    selection = {link_id: True for link_id in custom_links}
    _apply_custom_region_overrides(config, custom_links, log_run_id)
    calc_data = _dataset_to_calc_data(ds, custom_links, source_lookup, config=config, log_run_id=log_run_id)
    return SourceDataBundle(
        calc_data=calc_data,
        ips=_ips_from_links(custom_links),
        links=custom_links,
        selection=selection,
        realtime_buffer=None,
        time_min=_dataset_time_min(ds),
        time_max=_dataset_time_max(ds),
    )


def _load_pycomlink_dataset(config: dict[str, Any], mode: str) -> xr.Dataset:
    try:
        import pycomlink as pycml
    except Exception as exc:
        raise ProcessingException(
            "pycomlink is required for pycomlink custom dataset modes."
        ) from exc

    source_cfg = config.get("data_source", {})
    if mode == "pycomlink_example":
        data_dir = Path(pycml.io.examples.get_example_data_path())
        ds_path = data_dir / "example_cml_data.nc"
    else:
        ds_path = Path(str(source_cfg.get("dataset_path", "")))
        if not ds_path.exists():
            raise ProcessingException(f"Pycomlink NetCDF dataset not found: {ds_path}")

    ds = xr.open_dataset(ds_path)
    start, end = _config_time_slice(config)
    if "time" in ds.coords:
        ds = ds.sel(
            time=slice(
                np.datetime64(start.to_pydatetime()),
                np.datetime64(end.to_pydatetime()),
            )
        )
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("cml_id", 0) == 0:
        raise ProcessingException("Selected pycomlink dataset slice is empty.")
    return ds


def _load_openrainer_cml_dataset(*, config: dict[str, Any], log_run_id: str) -> xr.Dataset:
    source_cfg = config.get("data_source", {})
    root = Path(str(source_cfg.get("dataset_path", "")).strip())
    if not str(root):
        raise ProcessingException("[data_source].dataset_path must point to the OpenRainER dataset directory.")
    tar_path = root if root.is_file() else root / str(source_cfg.get("openrainer_cml_archive", "CML.tar")).strip()
    if not tar_path.exists():
        raise ProcessingException(f"OpenRainER CML archive not found: {tar_path}")

    start, end = _config_time_slice(config)
    member_names = _select_openrainer_cml_members(tar_path, start=start, end=end)
    if not member_names:
        raise ProcessingException(f"No OpenRainER CML monthly files overlap {start} .. {end} in {tar_path}.")

    logger.info(
        "[%s] Loading %d OpenRainER CML monthly NetCDF file(s) from %s.",
        log_run_id,
        len(member_names),
        tar_path,
    )

    datasets = [_open_netcdf_from_tar_gz(tar_path, member_name) for member_name in member_names]
    try:
        ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
        ds = ds.sortby("time")
        _, unique_idx = np.unique(ds.time.values, return_index=True)
        if len(unique_idx) != int(ds.time.size):
            ds = ds.isel(time=np.sort(unique_idx))
        ds = ds.sel(
            time=slice(
                np.datetime64(start.to_pydatetime()),
                np.datetime64(end.to_pydatetime()),
            )
        )
        rename_map = {
            "sublink_id": "channel_id",
            "site_0_lat": "site_a_latitude",
            "site_0_lon": "site_a_longitude",
            "site_1_lat": "site_b_latitude",
            "site_1_lon": "site_b_longitude",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords or k in ds.data_vars}
        if rename_map:
            ds = ds.rename(rename_map)
        if ds.sizes.get("time", 0) == 0 or ds.sizes.get("cml_id", 0) == 0:
            raise ProcessingException("Selected OpenRainER CML dataset slice is empty.")
        return ds
    finally:
        for child in datasets:
            try:
                child.close()
            except Exception:
                pass


def _select_openrainer_cml_members(
    tar_path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    names: list[tuple[pd.Timestamp, str]] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            match = _OPENRAINER_CML_MEMBER_RE.match(Path(member.name).name)
            if not match:
                continue
            member_start = pd.Timestamp(match.group(1), tz=None)
            member_end = pd.Timestamp(match.group(2), tz=None)
            if member_end >= start and member_start <= end:
                names.append((member_start, member.name))
    names.sort(key=lambda item: item[0])
    return [name for _, name in names]


def _select_openrainer_monthly_members(
    tar_path: Path,
    *,
    prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    names: list[tuple[pd.Timestamp, str]] = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            match = _OPENRAINER_MONTHLY_MEMBER_RE.match(base)
            if not match or match.group(1).lower() != prefix.lower():
                continue
            month_start = pd.Timestamp(f"{match.group(2)}01")
            month_end = month_start + pd.offsets.MonthEnd(1)
            if month_end >= start and month_start <= end:
                names.append((month_start, member.name))
    names.sort(key=lambda item: item[0])
    return [name for _, name in names]


def _open_netcdf_from_tar_gz(tar_path: Path, member_name: str) -> xr.Dataset:
    with tarfile.open(tar_path, "r") as tf:
        member = tf.getmember(member_name)
        extracted = tf.extractfile(member)
        if extracted is None:
            raise ProcessingException(f"Failed to extract {member_name} from {tar_path}")
        with gzip.GzipFile(fileobj=extracted) as gz:
            payload = gz.read()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    try:
        ds = xr.open_dataset(tmp_path).load()
        ds.close()
        return ds
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_openrainer_reference_dataset(
    *,
    config: dict[str, Any],
    source_name: str,
) -> xr.Dataset:
    source_cfg = config.get("data_source", {})
    root = Path(str(source_cfg.get("dataset_path", "")).strip())
    if not str(root):
        raise ProcessingException("[data_source].dataset_path must point to the OpenRainER dataset directory.")

    archive_name = {
        "radadj": "RADadj.tar",
        "radrain": "RADrain.tar",
        "radref": "RADref.tar",
        "aws": "AWS.tar",
    }.get(source_name.lower())
    if archive_name is None:
        raise ProcessingException(f"Unsupported OpenRainER reference source: {source_name}")
    tar_path = root / archive_name
    if not tar_path.exists():
        raise ProcessingException(f"OpenRainER reference archive not found: {tar_path}")

    start, end = _config_time_slice(config)
    member_names = _select_openrainer_monthly_members(tar_path, prefix=archive_name.split('.')[0], start=start, end=end)
    if not member_names:
        raise ProcessingException(f"No OpenRainER {source_name} monthly files overlap {start} .. {end} in {tar_path}.")

    datasets = [_open_netcdf_from_tar_gz(tar_path, member_name) for member_name in member_names]
    try:
        ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
        ds = ds.sortby("time")
        _, unique_idx = np.unique(ds.time.values, return_index=True)
        if len(unique_idx) != int(ds.time.size):
            ds = ds.isel(time=np.sort(unique_idx))
        ds = ds.sel(
            time=slice(
                np.datetime64(start.to_pydatetime()),
                np.datetime64(end.to_pydatetime()),
            )
        )
        return ds
    finally:
        for child in datasets:
            try:
                child.close()
            except Exception:
                pass


def _build_links_from_pycomlink_dataset(
    ds: xr.Dataset,
    *,
    min_length: float,
    max_length: float,
) -> tuple[dict[int, MwLink], dict[int, Any]]:
    links: dict[int, MwLink] = {}
    source_lookup: dict[int, Any] = {}
    for fallback_index, source_cml_id in enumerate(ds.cml_id.values, start=1):
        cml = ds.sel(cml_id=source_cml_id)
        lat_a = float(_scalar(cml, "site_a_latitude"))
        lon_a = float(_scalar(cml, "site_a_longitude"))
        lat_b = float(_scalar(cml, "site_b_latitude"))
        lon_b = float(_scalar(cml, "site_b_longitude"))
        length = float(_scalar(cml, "length"))
        if np.isfinite(length) and length > 1000.0:
            length /= 1000.0
        if not np.isfinite(length) or length <= 0:
            length = calc_distance(lat_a, lon_a, lat_b, lon_b)
        if length < min_length or length > max_length:
            continue

        link_id = _coerce_link_id(source_cml_id, fallback_index)
        freqs = _channel_array(cml, "frequency", default=[18.0, 18.0])
        pols = [_normalize_pol(v) for v in _channel_array(cml, "polarization", default=["V", "V"])]
        links[link_id] = MwLink(
            link_id=link_id,
            name=str(source_cml_id),
            tech="custom",
            name_a=f"{source_cml_id}_A",
            name_b=f"{source_cml_id}_B",
            freq_a=_freq_to_mhz(freqs[0]),
            freq_b=_freq_to_mhz(freqs[1] if len(freqs) > 1 else freqs[0]),
            polarization=pols[0] if pols else "V",
            ip_a=f"custom_{link_id}_a",
            ip_b=f"custom_{link_id}_b",
            distance=float(length),
            latitude_a=lat_a,
            longitude_a=lon_a,
            latitude_b=lat_b,
            longitude_b=lon_b,
            dummy_latitude_a=lat_a,
            dummy_longitude_a=lon_a,
            dummy_latitude_b=lat_b,
            dummy_longitude_b=lon_b,
        )
        source_lookup[link_id] = source_cml_id
    return links, source_lookup


def _dataset_to_calc_data(
    ds: xr.Dataset,
    links: dict[int, MwLink],
    source_lookup: dict[int, Any],
    *,
    config: dict[str, Any],
    log_run_id: str,
) -> list[xr.Dataset]:
    calc_data: list[xr.Dataset] = []
    resampled_count = 0
    native_minutes_seen: list[float] = []
    for link_id, source_cml_id in source_lookup.items():
        cml = ds.sel(cml_id=source_cml_id)
        tsl = _dataarray_2d(cml["tsl"]) if "tsl" in cml.data_vars else np.zeros_like(_dataarray_2d(cml["rsl"]))
        rsl = _dataarray_2d(cml["rsl"])
        channel_count = tsl.shape[0]
        channel_labels = (
            np.asarray(cml.channel_id.values, dtype=object)
            if "channel_id" in cml.coords and cml.channel_id.size == channel_count
            else np.asarray([f"channel_{i + 1}" for i in range(channel_count)], dtype=object)
        )
        freqs_ghz = _to_ghz(_channel_array(cml, "frequency", default=np.full(channel_count, 18.0)))
        pols = np.asarray([_normalize_pol(v) for v in _channel_array(cml, "polarization", default=["V"] * channel_count)], dtype=object)
        temps = np.zeros_like(rsl, dtype=float)

        link_ds = xr.Dataset(
            data_vars=dict(
                tsl=(("channel_id", "time"), tsl),
                rsl=(("channel_id", "time"), rsl),
                temperature_rx=(("channel_id", "time"), temps.copy()),
                temperature_tx=(("channel_id", "time"), temps.copy()),
            ),
            coords=dict(
                time=pd.to_datetime(cml.time.values).values.astype("datetime64[ns]"),
                channel_id=channel_labels,
                cml_id=link_id,
                site_a_latitude=float(_scalar(cml, "site_a_latitude")),
                site_b_latitude=float(_scalar(cml, "site_b_latitude")),
                site_a_longitude=float(_scalar(cml, "site_a_longitude")),
                site_b_longitude=float(_scalar(cml, "site_b_longitude")),
                frequency=("channel_id", freqs_ghz),
                polarization=("channel_id", pols),
                length=links[link_id].distance,
            ),
        )
        link_ds, was_resampled, native_minutes = _resample_calc_dataset_to_config_step(
            link_ds,
            config=config,
            log_run_id=log_run_id,
            link_id=link_id,
        )
        if native_minutes is not None:
            native_minutes_seen.append(float(native_minutes))
        if was_resampled:
            resampled_count += 1
        calc_data.append(link_ds)

    if resampled_count:
        step_min = int(config["time"]["step"])
        native_min = min(native_minutes_seen) if native_minutes_seen else float('nan')
        native_max = max(native_minutes_seen) if native_minutes_seen else float('nan')
        logger.info(
            "[%s] Resampled %d/%d custom links from native cadence %.2f..%.2f min to %d min base step.",
            log_run_id,
            resampled_count,
            len(calc_data),
            native_min,
            native_max,
            step_min,
        )
    return calc_data


def _resample_calc_dataset_to_config_step(
    link_ds: xr.Dataset,
    *,
    config: dict[str, Any],
    log_run_id: str,
    link_id: int,
) -> tuple[xr.Dataset, bool, Optional[float]]:
    if "time" not in link_ds.coords or link_ds.time.size == 0:
        return link_ds, False, None

    step_min = int(config["time"]["step"])
    if step_min <= 1:
        return link_ds, False, None

    native_times = pd.to_datetime(link_ds.time.values)
    if native_times.size < 2:
        return link_ds, False, None

    native_delta = native_times.to_series().diff().dropna().median()
    if pd.isna(native_delta):
        return link_ds, False, None

    native_minutes = float(native_delta / pd.Timedelta(minutes=1))
    if native_minutes >= step_min:
        return link_ds, False, native_minutes

    rule = f"{step_min}min"
    resampled = link_ds.resample(time=rule, label="right", closed="right").mean()

    rsl_valid = resampled["rsl"].notnull()
    tsl_valid = resampled["tsl"].notnull()
    for dim in [d for d in rsl_valid.dims if d != "time"]:
        rsl_valid = rsl_valid.any(dim=dim)
    for dim in [d for d in tsl_valid.dims if d != "time"]:
        tsl_valid = tsl_valid.any(dim=dim)
    valid_mask = np.asarray((rsl_valid | tsl_valid).values, dtype=bool)
    if valid_mask.any():
        resampled = resampled.isel(time=np.flatnonzero(valid_mask))

    return resampled, True, native_minutes


_NETHERLANDS_RAW_DATE_RE = re.compile(r"^[A-Za-z0-9]+_(\d{1,2})-(\d{1,2})-(\d{4})\.csv\.gz$", re.IGNORECASE)
_OPENRAINER_CML_MEMBER_RE = re.compile(r"^CML_(\d{12})_(\d{12})\.nc\.gz$", re.IGNORECASE)
_OPENRAINER_MONTHLY_MEMBER_RE = re.compile(r"^(AWS|RADadj|RADrain|RADref)_(\d{6})(?:\d{6})?\.nc\.gz$", re.IGNORECASE)


def _load_netherlands_raw_calc_data(
    *,
    config: dict[str, Any],
    log_run_id: str,
) -> tuple[dict[int, MwLink], list[xr.Dataset], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    source_cfg = config.get("data_source", {})
    dataset_path = Path(str(source_cfg.get("dataset_path", "")).strip())
    if not str(dataset_path):
        raise ProcessingException(
            "[data_source].dataset_path must point to the extracted Netherlands RawCMLdata directory."
        )
    if not dataset_path.exists():
        raise ProcessingException(f"Netherlands raw dataset path not found: {dataset_path}")

    root = _resolve_netherlands_raw_root(dataset_path)
    start, end = _config_time_slice(config)
    file_glob = str(source_cfg.get("netherlands_file_glob", "NEC_*.csv.gz")).strip() or "NEC_*.csv.gz"
    signal_stat = str(source_cfg.get("netherlands_signal_stat", "rxmin")).strip().lower()
    if signal_stat not in {"rxmin", "rxmax", "mid"}:
        raise ProcessingException(
            "[data_source].netherlands_signal_stat must be one of: rxmin, rxmax, mid."
        )
    default_pol = _normalize_pol(source_cfg.get("netherlands_polarization", "V"))
    mask_errored = bool(source_cfg.get("netherlands_mask_errored", True))

    files = _select_netherlands_raw_files(root, start=start, end=end, file_glob=file_glob)
    if not files:
        raise ProcessingException(
            f"No Netherlands raw CSV files found for {start} .. {end} under {root}."
        )

    logger.info(
        "[%s] Loading %d Netherlands raw CSV file(s) from %s using signal_stat=%s.",
        log_run_id,
        len(files),
        root,
        signal_stat,
    )

    records_by_name: dict[str, list[pd.DataFrame]] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    time_min: Optional[pd.Timestamp] = None
    time_max: Optional[pd.Timestamp] = None

    for file_path in files:
        df = pd.read_csv(
            file_path,
            compression="gzip",
            usecols=[
                "LINK_ID",
                "SITE_ID",
                "SITE_LAT_SECS",
                "SITE_LON_SECS",
                "FAR_END_SITE_ID",
                "FAR_END_LAT_SECS",
                "FAR_END_LON_SECS",
                "FREQ",
                "YYYYMMDDHHMMSS",
                "DURATION",
                "ES",
                "SES",
                "RXMIN_1",
                "RXMAX_1",
            ],
            dtype=str,
        )
        if df.empty:
            continue

        df["time"] = pd.to_datetime(df["YYYYMMDDHHMMSS"], format="%Y%m%d%H%M%S", errors="coerce")
        df = df.dropna(subset=["time", "LINK_ID"])
        if df.empty:
            continue
        df = df.loc[(df["time"] >= start) & (df["time"] <= end)].copy()
        if df.empty:
            continue

        rxmin = pd.to_numeric(df["RXMIN_1"], errors="coerce")
        rxmax = pd.to_numeric(df["RXMAX_1"], errors="coerce")
        if signal_stat == "rxmax":
            df["rsl"] = rxmax.combine_first(rxmin)
        elif signal_stat == "mid":
            df["rsl"] = pd.concat([rxmin, rxmax], axis=1).mean(axis=1, skipna=True)
        else:
            df["rsl"] = rxmin.combine_first(rxmax)

        if mask_errored:
            es = pd.to_numeric(df["ES"], errors="coerce").fillna(0.0)
            ses = pd.to_numeric(df["SES"], errors="coerce").fillna(0.0)
            df.loc[(es > 0.0) | (ses > 0.0), "rsl"] = np.nan

        cur_min = pd.Timestamp(df["time"].min())
        cur_max = pd.Timestamp(df["time"].max())
        time_min = cur_min if time_min is None else min(time_min, cur_min)
        time_max = cur_max if time_max is None else max(time_max, cur_max)

        for link_name, group in df.groupby("LINK_ID", sort=False):
            series = group.loc[:, ["time", "rsl"]].copy()
            records_by_name.setdefault(str(link_name), []).append(series)
            if str(link_name) not in meta_by_name:
                meta_by_name[str(link_name)] = group.iloc[0].to_dict()

    if not records_by_name:
        raise ProcessingException(
            f"Selected Netherlands raw dataset window is empty after filtering: {start} .. {end}."
        )

    links: dict[int, MwLink] = {}
    calc_data: list[xr.Dataset] = []
    resampled_count = 0
    native_minutes_seen: list[float] = []
    min_length = float(config["cml"]["min_length"])
    max_length = float(config["cml"]["max_length"])

    for fallback_index, link_name in enumerate(sorted(records_by_name), start=1):
        meta = meta_by_name[link_name]
        lat_a = _netherlands_coord_to_deg(meta.get("SITE_LAT_SECS"))
        lon_a = _netherlands_coord_to_deg(meta.get("SITE_LON_SECS"))
        lat_b = _netherlands_coord_to_deg(meta.get("FAR_END_LAT_SECS"))
        lon_b = _netherlands_coord_to_deg(meta.get("FAR_END_LON_SECS"))
        length = calc_distance(lat_a, lon_a, lat_b, lon_b)
        if not np.isfinite(length) or length < min_length or length > max_length:
            continue

        link_id = fallback_index
        freq_mhz = _freq_to_mhz(meta.get("FREQ", 18000.0))
        link = MwLink(
            link_id=link_id,
            name=str(link_name),
            tech="netherlands_raw",
            name_a=str(meta.get("SITE_ID", f"{link_name}_A")),
            name_b=str(meta.get("FAR_END_SITE_ID", f"{link_name}_B")),
            freq_a=freq_mhz,
            freq_b=freq_mhz,
            polarization=default_pol,
            ip_a=f"custom_{link_id}_a",
            ip_b=f"custom_{link_id}_b",
            distance=float(length),
            latitude_a=lat_a,
            longitude_a=lon_a,
            latitude_b=lat_b,
            longitude_b=lon_b,
            dummy_latitude_a=lat_a,
            dummy_longitude_a=lon_a,
            dummy_latitude_b=lat_b,
            dummy_longitude_b=lon_b,
        )

        series = pd.concat(records_by_name[link_name], ignore_index=True)
        series = series.sort_values("time").drop_duplicates(subset="time", keep="last")
        if series.empty or not series["rsl"].notna().any():
            continue

        rsl = np.asarray(series["rsl"].to_numpy(dtype=float), dtype=float).reshape(1, -1)
        zeros = np.zeros_like(rsl, dtype=float)
        link_ds = xr.Dataset(
            data_vars=dict(
                tsl=(("channel_id", "time"), zeros.copy()),
                rsl=(("channel_id", "time"), rsl),
                temperature_rx=(("channel_id", "time"), zeros.copy()),
                temperature_tx=(("channel_id", "time"), zeros.copy()),
            ),
            coords=dict(
                time=pd.to_datetime(series["time"]).values.astype("datetime64[ns]"),
                channel_id=np.asarray(["channel_1"], dtype=object),
                cml_id=link_id,
                site_a_latitude=float(lat_a),
                site_b_latitude=float(lat_b),
                site_a_longitude=float(lon_a),
                site_b_longitude=float(lon_b),
                frequency=("channel_id", np.asarray([float(freq_mhz) / 1000.0], dtype=float)),
                polarization=("channel_id", np.asarray([default_pol], dtype=object)),
                length=float(length),
            ),
        )
        link_ds, was_resampled, native_minutes = _resample_calc_dataset_to_config_step(
            link_ds,
            config=config,
            log_run_id=log_run_id,
            link_id=link_id,
        )
        if native_minutes is not None:
            native_minutes_seen.append(float(native_minutes))
        if was_resampled:
            resampled_count += 1

        links[link_id] = link
        calc_data.append(link_ds)

    if not calc_data:
        raise ProcessingException(
            f"No usable Netherlands links remained after filtering for {start} .. {end}."
        )

    if resampled_count:
        step_min = int(config["time"]["step"])
        native_min = min(native_minutes_seen) if native_minutes_seen else float("nan")
        native_max = max(native_minutes_seen) if native_minutes_seen else float("nan")
        logger.info(
            "[%s] Resampled %d/%d Netherlands links from native cadence %.2f..%.2f min to %d min base step.",
            log_run_id,
            resampled_count,
            len(calc_data),
            native_min,
            native_max,
            step_min,
        )

    logger.info(
        "[%s] Built %d Netherlands raw CML dataset(s) from %d day file(s).",
        log_run_id,
        len(calc_data),
        len(files),
    )
    return links, calc_data, time_min, time_max


def _resolve_netherlands_raw_root(path: Path) -> Path:
    candidates = [path, path / "RawCMLdata"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if any((candidate / str(year)).exists() for year in range(2011, 2016)):
            return candidate
        if list(candidate.rglob("*.csv.gz")):
            return candidate
    raise ProcessingException(
        f"Could not resolve Netherlands raw dataset root from {path}. Expected year folders or CSV.GZ files."
    )


def _select_netherlands_raw_files(
    root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    file_glob: str,
) -> list[Path]:
    start_date = start.date()
    end_date = end.date()
    files: list[tuple[pd.Timestamp, Path]] = []
    for year in range(start.year, end.year + 1):
        year_dir = root / str(year)
        if not year_dir.exists():
            continue
        for file_path in year_dir.glob(file_glob):
            file_date = _netherlands_date_from_filename(file_path.name)
            if file_date is None:
                continue
            if start_date <= file_date.date() <= end_date:
                files.append((file_date, file_path))
    if not files and root.exists():
        for file_path in root.rglob(file_glob):
            file_date = _netherlands_date_from_filename(file_path.name)
            if file_date is None:
                continue
            if start_date <= file_date.date() <= end_date:
                files.append((file_date, file_path))
    files.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in files]


def _netherlands_date_from_filename(name: str) -> Optional[pd.Timestamp]:
    match = _NETHERLANDS_RAW_DATE_RE.match(name)
    if not match:
        return None
    day, month, year = (int(match.group(i)) for i in range(1, 4))
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def _netherlands_coord_to_deg(value: Any) -> float:
    return float(value) / 3600000.0


def _apply_custom_region_overrides(
    config: dict[str, Any],
    links: dict[int, MwLink],
    log_run_id: str,
) -> None:
    source_cfg = config.get("data_source", {})
    auto_adjust = bool(source_cfg.get("auto_adjust_region", True))
    padding = float(source_cfg.get("region_padding_deg", 0.3))

    if auto_adjust and links:
        lons = []
        lats = []
        for link in links.values():
            lons.extend([float(link.longitude_a), float(link.longitude_b)])
            lats.extend([float(link.latitude_a), float(link.latitude_b)])
        config.setdefault("limits", {})
        config["limits"]["x_min"] = min(lons) - padding
        config["limits"]["x_max"] = max(lons) + padding
        config["limits"]["y_min"] = min(lats) - padding
        config["limits"]["y_max"] = max(lats) + padding
        logger.info(
            "[%s] Auto-adjusted [limits] to custom dataset extent with %.3f degree padding.",
            log_run_id,
            padding,
        )



def _config_time_slice(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(config["time"]["start"])
    end = pd.Timestamp(config["time"]["end"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    return start, end


def _scalar(ds: xr.Dataset, name: str):
    if name in ds.coords:
        value = ds.coords[name].values
    elif name in ds.data_vars:
        value = ds[name].values
    else:
        raise ProcessingException(f"Missing required field {name!r} in custom dataset.")
    arr = np.asarray(value)
    return arr.reshape(-1)[0]


def _channel_array(ds: xr.Dataset, name: str, default) -> np.ndarray:
    if name in ds.coords:
        value = ds.coords[name].values
    elif name in ds.data_vars:
        value = ds[name].values
    else:
        return np.asarray(default)
    arr = np.asarray(value)
    return np.asarray([arr.item()]) if arr.ndim == 0 else arr.reshape(-1)


def _dataarray_2d(arr: xr.DataArray) -> np.ndarray:
    if "channel_id" not in arr.dims:
        arr = arr.expand_dims(channel_id=["channel_1"])
    return arr.transpose("channel_id", "time").values.astype(float)


def _to_ghz(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, 18.0, dtype=float)
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs > 1e6:
        return arr / 1e9
    if max_abs > 100.0:
        return arr / 1000.0
    return arr


def _freq_to_mhz(value: Any) -> int:
    f = float(value)
    if abs(f) > 1e8:
        f /= 1e6
    elif abs(f) > 1e5:
        f /= 1e3
    elif abs(f) < 100.0:
        f *= 1000.0
    return int(round(f))


def _normalize_pol(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    s = str(value).strip().upper()
    if s.startswith("X"):
        return "V"
    return s if s in {"H", "V"} else "V"


def _coerce_link_id(source_id: Any, fallback_index: int) -> int:
    try:
        return int(source_id)
    except (TypeError, ValueError):
        return int(fallback_index)


def _ips_from_links(links: dict[int, MwLink]) -> list[str]:
    ips: list[str] = []
    for link in links.values():
        ips.extend([link.ip_a, link.ip_b])
    return sorted(set(ips))


def export_custom_cml_metadata_json(
    *,
    config: dict[str, Any],
    links: dict[int, MwLink],
    calc_data: Optional[xr.Dataset | list[xr.Dataset]],
    log_run_id: str = "custom",
) -> Optional[Path]:
    source_cfg = config.get("data_source", {})
    if not bool(source_cfg.get("export_cml_metadata_json", False)):
        return None

    out_dir = Path(str(config["directories"]["outputs_json"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = str(source_cfg.get("cml_metadata_json_filename", "cml_metadata.json")).strip() or "cml_metadata.json"
    out_path = out_dir / filename

    calc_by_id: dict[int, xr.Dataset] = {}
    if isinstance(calc_data, list):
        for ds in calc_data:
            try:
                calc_by_id[int(ds.cml_id.values)] = ds
            except Exception:
                continue
    elif calc_data is not None and "cml_id" in calc_data.coords:
        for idx in range(int(calc_data.cml_id.size)):
            ds = calc_data.isel(cml_id=idx)
            try:
                calc_by_id[int(ds.cml_id.values)] = ds
            except Exception:
                continue

    items = []
    for link_id in sorted(links):
        link = links[link_id]
        ds = calc_by_id.get(int(link_id))
        frequencies_ghz = []
        polarizations = []
        channel_ids = []
        if ds is not None:
            if "frequency" in ds.coords:
                frequencies_ghz = [float(v) for v in np.asarray(ds.frequency.values).reshape(-1)]
            if "polarization" in ds.coords:
                polarizations = [str(v) for v in np.asarray(ds.polarization.values).reshape(-1)]
            if "channel_id" in ds.coords:
                channel_ids = [str(v) for v in np.asarray(ds.channel_id.values).reshape(-1)]
        if not frequencies_ghz:
            frequencies_ghz = [float(link.freq_a) / 1000.0, float(link.freq_b) / 1000.0]
        if not polarizations:
            polarizations = [str(link.polarization), str(link.polarization)]
        if not channel_ids:
            channel_ids = ["channel_1", "channel_2"]

        item = {
            "cml_id": int(link.link_id),
            "name": str(link.name),
            "tech": str(link.tech),
            "site_a": {
                "name": str(link.name_a),
                "latitude": float(link.latitude_a),
                "longitude": float(link.longitude_a),
                "ip": str(link.ip_a),
            },
            "site_b": {
                "name": str(link.name_b),
                "latitude": float(link.latitude_b),
                "longitude": float(link.longitude_b),
                "ip": str(link.ip_b),
            },
            "length_km": float(link.distance),
            "polarization": str(link.polarization),
            "freq_a_mhz": int(link.freq_a),
            "freq_b_mhz": int(link.freq_b),
            "channel_ids": channel_ids,
            "frequencies_ghz": frequencies_ghz,
            "polarizations": polarizations,
        }
        items.append(item)

    payload = {
        "data_source_mode": get_data_source_mode(config),
        "count": len(items),
        "limits": {
            "x_min": float(config["limits"]["x_min"]),
            "x_max": float(config["limits"]["x_max"]),
            "y_min": float(config["limits"]["y_min"]),
            "y_max": float(config["limits"]["y_max"]),
        },
        "cmls": items,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info("[%s] Exported custom CML metadata JSON to %s", log_run_id, out_path)
    return out_path


def export_custom_rain_event_summary_csv(
    *,
    config: dict[str, Any],
    calc_dataset: Optional[xr.Dataset],
    log_run_id: str = "custom",
) -> Optional[Path]:
    source_cfg = config.get("data_source", {})
    if not bool(source_cfg.get("export_rain_event_summary_csv", False)):
        return None
    if calc_dataset is None or "time" not in calc_dataset.dims or calc_dataset.time.size == 0:
        return None
    if "R" not in calc_dataset:
        return None

    out_dir = Path(str(config["directories"]["outputs_json"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        str(source_cfg.get("rain_event_summary_filename", "rain_event_summary.csv")).strip()
        or "rain_event_summary.csv"
    )
    out_path = out_dir / filename
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utc",
                "rainy_link_count",
                "total_link_count",
                "rainy_link_fraction",
                "max_link_rain_mm_h",
                "mean_link_rain_mm_h",
                "p95_link_rain_mm_h",
            ],
        )
        if write_header:
            writer.writeheader()

        for t in range(int(calc_dataset.time.size)):
            sl = calc_dataset.isel(time=t)
            r = np.asarray(sl["R"].values, dtype=float)
            if r.ndim == 0:
                link_mean = np.asarray([float(r)])
            elif r.ndim == 1:
                link_mean = r
            else:
                valid_counts = np.sum(np.isfinite(r), axis=1)
                sums = np.nansum(r, axis=1)
                link_mean = np.full(valid_counts.shape, np.nan, dtype=float)
                nonempty = valid_counts > 0
                link_mean[nonempty] = sums[nonempty] / valid_counts[nonempty]

            finite = np.isfinite(link_mean)
            rainy = finite & (link_mean > 0.0)
            rainy_count = int(np.sum(rainy))
            total_count = int(link_mean.shape[0])

            if finite.any():
                finite_vals = link_mean[finite]
                max_rain = float(np.nanmax(finite_vals))
                mean_rain = float(np.nanmean(finite_vals))
                p95_rain = float(np.nanpercentile(finite_vals, 95))
            else:
                max_rain = 0.0
                mean_rain = 0.0
                p95_rain = 0.0

            ts = pd.Timestamp(sl.time.values)
            writer.writerow(
                {
                    "utc": ts.strftime("%Y-%m-%d_%H%M"),
                    "rainy_link_count": rainy_count,
                    "total_link_count": total_count,
                    "rainy_link_fraction": (float(rainy_count) / float(total_count)) if total_count else 0.0,
                    "max_link_rain_mm_h": max_rain,
                    "mean_link_rain_mm_h": mean_rain,
                    "p95_link_rain_mm_h": p95_rain,
                }
            )

    logger.info("[%s] Appended rain-event summary CSV rows to %s", log_run_id, out_path)
    return out_path


def export_openrainer_reference_pngs(
    *,
    config: dict[str, Any],
    log_run_id: str = "custom",
) -> Optional[Path]:
    if get_data_source_mode(config) != "openrainer_tar":
        return None

    source_cfg = config.get("data_source", {})
    if not bool(source_cfg.get("openrainer_export_reference_pngs", True)):
        return None
    ref_source = str(source_cfg.get("openrainer_reference_source", "radadj")).strip().lower()
    if ref_source in {"", "none"}:
        return None
    if ref_source not in {"radadj", "radrain"}:
        logger.info("[%s] Skipping OpenRainER reference PNG export for unsupported source=%s.", log_run_id, ref_source)
        return None

    ds = _load_openrainer_reference_dataset(config=config, source_name=ref_source)
    if "rainfall_amount" not in ds.data_vars or "lat" not in ds.coords or "lon" not in ds.coords:
        raise ProcessingException(f"OpenRainER {ref_source} dataset does not expose rainfall_amount on lat/lon grid.")

    limits = config.get("limits", {})
    x_min = float(limits["x_min"])
    x_max = float(limits["x_max"])
    y_min = float(limits["y_min"])
    y_max = float(limits["y_max"])

    ds = ds.sel(lon=slice(x_min, x_max), lat=slice(y_min, y_max))
    if ds.sizes.get("time", 0) == 0 or ds.sizes.get("lat", 0) == 0 or ds.sizes.get("lon", 0) == 0:
        raise ProcessingException("Selected OpenRainER reference slice is empty after applying time/bbox selection.")

    if float(ds.lat.values[0]) < float(ds.lat.values[-1]):
        ds = ds.isel(lat=slice(None, None, -1))

    lat_vals = np.asarray(ds.lat.values, dtype=float)
    lon_vals = np.asarray(ds.lon.values, dtype=float)
    static_mask = _build_latlon_crop_mask(config=config, lat_vals=lat_vals, lon_vals=lon_vals)

    out_dir = Path(str(config.get("directories", {}).get("outputs_reference_web", "outputs_reference_web")))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(str(config.get("directories", {}).get("outputs_reference_json", "outputs_reference_json")))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = str(source_cfg.get("openrainer_reference_manifest_filename", "openrainer_reference_manifest.json")).strip() or "openrainer_reference_manifest.json"
    manifest_path = manifest_dir / manifest_name

    written = 0
    for t in range(int(ds.time.size)):
        sl = ds.isel(time=t)
        grid = np.asarray(sl["rainfall_amount"].values, dtype=float)
        # RADadj/RADrain are 15 min accumulations; convert to mm/h equivalent for the same rain palette.
        grid = grid * 4.0
        if static_mask is not None:
            grid = np.where(static_mask, grid, np.nan)
        rgba = _rain_grid_to_rgba(grid)
        ts = pd.Timestamp(sl.time.values)
        fname = ts.strftime("%Y-%m-%d_%H%M")
        from PIL import Image
        Image.fromarray(rgba, "RGBA").save(out_dir / f"{fname}.png")
        written += 1

    payload = {
        "source": ref_source,
        "value_units": "mm_h_equivalent",
        "x_min": float(lon_vals.min()),
        "x_max": float(lon_vals.max()),
        "y_min": float(lat_vals.min()),
        "y_max": float(lat_vals.max()),
        "count": int(written),
        "time_min": pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d_%H%M"),
        "time_max": pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d_%H%M"),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[%s] Exported %d OpenRainER %s reference PNG(s) to %s", log_run_id, written, ref_source, out_dir)
    return manifest_path


def _rain_grid_to_rgba(grid: np.ndarray) -> np.ndarray:
    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)
    finite = np.isfinite(grid)
    if not finite.any():
        return rgba
    vals = grid[finite]
    idx = np.searchsorted(RAIN_THRESH, vals, side="right") - 1
    valid = idx >= 0
    if np.any(valid):
        rgba_vals = np.zeros((vals.shape[0], 4), dtype=np.uint8)
        rgba_vals[valid] = RAIN_COLORS[idx[valid]]
        rgba[finite] = rgba_vals
    return rgba


def _build_latlon_crop_mask(
    *,
    config: dict[str, Any],
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> Optional[np.ndarray]:
    rendering_cfg = config.get("rendering", {})
    if not bool(rendering_cfg.get("is_crop_enabled", False)):
        return None
    geojson_name = str(rendering_cfg.get("geojson_file", "")).strip()
    if not geojson_name:
        return None

    geojson_path = Path("./assets") / geojson_name
    if not geojson_path.exists():
        raise ProcessingException(f"Crop GeoJSON file not found: {geojson_path}")

    from shapely.geometry import Point as GeoPoint, shape
    from shapely.ops import unary_union
    from shapely.prepared import prep

    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    polys = [shape(feat["geometry"]).buffer(0) for feat in data.get("features", []) if feat.get("geometry")]
    if not polys:
        return None
    merged = unary_union(polys).buffer(0)
    prep_poly = prep(merged)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    bbox = merged.bounds
    bbox_mask = (
        (lon_grid >= bbox[0]) & (lon_grid <= bbox[2]) & (lat_grid >= bbox[1]) & (lat_grid <= bbox[3])
    )
    mask = np.zeros_like(bbox_mask, dtype=bool)
    pts = np.column_stack((lon_grid[bbox_mask], lat_grid[bbox_mask]))
    inside = [prep_poly.contains(GeoPoint(x, y)) for x, y in pts]
    mask[bbox_mask] = inside
    return mask


def _df_time_min(df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "_time" not in df.columns:
        return None
    return pd.to_datetime(df["_time"], utc=True).min()


def _df_time_max(df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "_time" not in df.columns:
        return None
    return pd.to_datetime(df["_time"], utc=True).max()


def _dataset_time_min(ds: xr.Dataset) -> Optional[pd.Timestamp]:
    if "time" not in ds.coords or ds.time.size == 0:
        return None
    return pd.Timestamp(pd.to_datetime(ds.time.values).min())


def _dataset_time_max(ds: xr.Dataset) -> Optional[pd.Timestamp]:
    if "time" not in ds.coords or ds.time.size == 0:
        return None
    return pd.Timestamp(pd.to_datetime(ds.time.values).max())
