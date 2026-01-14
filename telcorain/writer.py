import glob
import json
import math
import os
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

import numpy as np
from influxdb_client import Point, WritePrecision
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point as GeoPoint
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union
from shapely.prepared import prep
from xarray import Dataset

from telcorain.cython.raincolor import rain_to_rgba
from telcorain.handlers import logger
from telcorain.helpers import (
    dt64_to_unixtime,
    save_ndarray_to_file,
    _hex_to_rgba_u8,
    rain_to_rgba_custom,
    get_rain_sum_colors,
    verify_hour_sum,
)


class Writer:
    def __init__(
        self,
        influx_man,
        skip_influx: bool,
        config: dict,
        since_time: Optional[datetime] = None,
        is_historic: bool = False,
        is_web: bool = False,
        influx_wipe_thread: Optional[Thread] = None,
    ):
        self.influx_man = influx_man
        self.skip_influx = skip_influx  # controls NORMAL 10-min influx writing only
        self.config = config
        self.is_historic = is_historic
        self.influx_wipe_thread = influx_wipe_thread

        self.is_crop_enabled = self.config["rendering"]["is_crop_enabled"]
        self.geojson_file = self.config["rendering"]["geojson_file"]
        self.output_json_info = self.config["setting"]["output_json_info"]

        # output dirs
        if is_historic or is_web:
            user_dir = self.config["user_info"]["folder_name"]
            output_json_dir = self.config["directories"]["outputs_json"]
            output_sum_dir = self.config["directories"]["outputs_sum"]
            output_sum_json_dir = self.config["directories"]["outputs_sum_json"]

            self.outputs_raw_dir = f"{user_dir}/outputs_raw"
            self.outputs_web_dir = f"{user_dir}/outputs_web"
            self.outputs_json_dir = f"{user_dir}/{output_json_dir}"
            self.outputs_sum_dir = f"{user_dir}/{output_sum_dir}"
            self.outputs_sum_json_dir = f"{user_dir}/{output_sum_json_dir}"
        else:
            self.outputs_raw_dir = self.config["directories"]["outputs_raw"]
            self.outputs_web_dir = self.config["directories"]["outputs_web"]
            self.outputs_json_dir = self.config["directories"]["outputs_json"]
            self.outputs_sum_dir = self.config["directories"]["outputs_sum"]
            self.outputs_sum_json_dir = self.config["directories"]["outputs_sum_json"]

        # overall intensity parameters from config
        self.overall_intensity_ref = self.config["raingrids"]["overall_intensity_ref"]
        self.overall_intensity_method = self.config["raingrids"][
            "overall_intensity_method"
        ]
        self.overall_intensity_threshold = self.config["raingrids"][
            "overall_intensity_threshold"
        ]
        self.overall_intensity_coverage_gamma = self.config["raingrids"][
            "overall_intensity_coverage_gamma"
        ]
        self.overall_intensity_strength_gamma = self.config["raingrids"][
            "overall_intensity_strength_gamma"
        ]
        self.min_if_any = self.config["raingrids"]["overall_intensity_min_if_any_link"]

        if since_time is None:
            since_time = datetime.min
        self.since_time = since_time

        # Mask cache
        self.static_mask = None
        self.prep_poly = None
        self.bbox = None

        # CRS logic
        self.use_mercator = bool(self.config["interp"].get("use_mercator", False))
        if self.use_mercator:
            self.transform_fwd = Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            )
            self.transform_back = Transformer.from_crs(
                "EPSG:3857", "EPSG:4326", always_xy=True
            )
        else:
            self.transform_fwd = None
            self.transform_back = None

        self.hs_palette = get_rain_sum_colors()
        levels_sorted = sorted(self.hs_palette.keys())
        cols_sorted = [self.hs_palette[k] for k in levels_sorted]
        self.hs_levels = np.asarray(levels_sorted, dtype=float)
        self.hs_colors = np.stack([_hex_to_rgba_u8(c) for c in cols_sorted], axis=0)

    # ------------------------------------------------------------------
    # POLYGON MASKING
    # ------------------------------------------------------------------

    def _load_polygon(self):
        """Load GeoJSON mask, reproject to Mercator if needed."""
        with open(f"./assets/{self.geojson_file}", "r", encoding="utf-8") as f:
            data = json.load(f)

        polys = [shape(feat["geometry"]).buffer(0) for feat in data["features"]]

        if self.use_mercator:

            def proj(x, y, z=None):
                return self.transform_fwd.transform(x, y)

            polys = [shp_transform(proj, p) for p in polys]

        merged = unary_union(polys).buffer(0)
        return prep(merged), merged.bounds

    def _compute_static_mask(self, x_grid, y_grid):
        minx, miny, maxx, maxy = self.bbox
        bbox_mask = (
            (x_grid >= minx) & (x_grid <= maxx) & (y_grid >= miny) & (y_grid <= maxy)
        )

        static_mask = np.zeros_like(bbox_mask, dtype=bool)
        pts = np.column_stack((x_grid[bbox_mask], y_grid[bbox_mask]))

        inside = [self.prep_poly.contains(GeoPoint(x, y)) for x, y in pts]
        static_mask[bbox_mask] = inside
        return static_mask

    # ------------------------------------------------------------------
    # PNG WRITER
    # ------------------------------------------------------------------

    def _compute_overall_intensity(
        self,
        grid: np.ndarray,
        mode: str = "intensity",
        window_minutes: float = 60.0,
    ) -> float:
        """
        Compute a single 0..1 'overall' score using the SAME [raingrids] parameters.

        mode="intensity": grid is mm/h (no conversion)
        mode="hour_sum":  grid is mm accumulated over `window_minutes`,
                        converted internally to mm/h so existing thresholds/ref remain valid.
        """
        if grid is None:
            return 0.0

        valid = np.asarray(grid, dtype=float)
        if valid.size == 0:
            return 0.0

        # Convert hour-sum (mm over window) to mm/h so we can reuse the same scoring params
        if mode == "hour_sum":
            win_h = float(window_minutes) / 60.0
            if win_h <= 0:
                win_h = 1.0
            valid = valid / win_h  # mm -> mm/h equivalent

        method = str(self.overall_intensity_method).lower()
        threshold = float(self.overall_intensity_threshold)
        ref = float(self.overall_intensity_ref)
        cg = float(self.overall_intensity_coverage_gamma)
        sg = float(self.overall_intensity_strength_gamma)

        if method in {"coverage", "coverage_strength"}:
            finite = np.isfinite(valid)
            denom = int(np.count_nonzero(finite))
            if denom == 0:
                return 0.0

            wet = finite & (valid > threshold)
            coverage = float(np.count_nonzero(wet)) / float(denom)

            if method == "coverage":
                return float(min(1.0, max(0.0, coverage)))

            wet_vals = valid[wet]
            if wet_vals.size == 0:
                return 0.0

            stat = float(np.nanmean(wet_vals))
            if not math.isfinite(stat) or stat <= 0:
                return 0.0

            ref = max(ref, 1e-6)
            strength = math.log1p(stat) / math.log1p(ref)
            strength = float(min(1.0, max(0.0, strength)))

            cg = max(cg, 1e-6)
            sg = max(sg, 1e-6)
            score = (coverage**cg) * (strength**sg)
            return float(min(1.0, max(0.0, score)))

        if method == "p90":
            stat = float(np.nanpercentile(valid, 90))
        elif method == "p95":
            stat = float(np.nanpercentile(valid, 95))
        else:
            stat = float(np.nanmean(valid))

        if not math.isfinite(stat) or stat <= 0:
            return 0.0

        ref = max(ref, 1e-6)
        score = math.log1p(stat) / math.log1p(ref)
        return float(min(1.0, max(0.0, score)))

    def _write_raingrids(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ):
        if self.is_crop_enabled and self.static_mask is None:
            self.prep_poly, self.bbox = self._load_polygon()
            self.static_mask = self._compute_static_mask(x_grid, y_grid)

        if self.use_mercator and self.config["logging"]["init_level"] == "debug":
            minX, minY = x_grid.min(), y_grid.min()
            maxX, maxY = x_grid.max(), y_grid.max()
            min_lon, min_lat = self.transform_back.transform(minX, minY)
            max_lon, max_lat = self.transform_back.transform(maxX, maxY)
            logger.debug(
                "[WGS84] Data for OSM overlay: "
                f"min_lon={min_lon}, min_lat={min_lat}, "
                f"max_lon={max_lon}, max_lat={max_lat}"
            )

        for t in range(len(calc_dataset.time)):
            time_val = calc_dataset.time[t]
            dt = datetime.utcfromtimestamp(dt64_to_unixtime(time_val.values))
            fname = dt.strftime("%Y-%m-%d_%H%M")

            raw_path = f"{self.outputs_raw_dir}/{fname}.npy"

            existing_png = glob.glob(f"{self.outputs_web_dir}/{fname}_*.png")
            existing_png += glob.glob(f"{self.outputs_web_dir}/{fname}.png")

            need_raw = self.config["directories"]["save_raw"] and not os.path.exists(
                raw_path
            )
            need_png = self.config["directories"]["save_web"] and (
                len(existing_png) == 0
            )
            if not (need_raw or need_png):
                continue

            grid = rain_grids[t]

            if self.is_crop_enabled:
                grid = np.where(self.static_mask, grid, np.nan)

            overall = self._compute_overall_intensity(grid)

            R = calc_dataset["R"].isel(time=t).values
            any_link_wet = bool((R > 0).any())
            if any_link_wet:
                overall = max(overall, self.min_if_any)
            overall_str = f"{overall:.3f}"

            png_path = f"{self.outputs_web_dir}/{fname}_{overall_str}.png"

            if need_raw:
                save_ndarray_to_file(grid, raw_path)

            if need_png:
                rgba = rain_to_rgba(grid)
                rgba = np.flipud(rgba)
                Image.fromarray(rgba, "RGBA").save(png_path)

            if self.output_json_info:
                self._write_json_rain_flags(calc_dataset, t, fname, overall)

        logger.info("[WRITE] PNGs saved.")

    def _write_sum_raingrids(self, rain_grids_sum, x_grid, y_grid, calc_dataset):
        if rain_grids_sum is None:
            return

        os.makedirs(self.outputs_sum_dir, exist_ok=True)

        if self.is_crop_enabled and self.static_mask is None:
            self.prep_poly, self.bbox = self._load_polygon()
            self.static_mask = self._compute_static_mask(x_grid, y_grid)

        n = min(len(calc_dataset.time), len(rain_grids_sum))
        if n == 0:
            return

        if len(rain_grids_sum) != len(calc_dataset.time):
            logger.warning(
                f"[WRITE] Hour-sum length mismatch: grids_sum={len(rain_grids_sum)} "
                f"vs time={len(calc_dataset.time)}. Writing last {n} frames."
            )

        time_slice = calc_dataset.isel(time=slice(-n, None))

        for i in range(n):
            t = -n + i
            time_val = calc_dataset.time[t]
            dt = datetime.utcfromtimestamp(dt64_to_unixtime(time_val.values))
            fname = dt.strftime("%Y-%m-%d_%H%M")

            grid = rain_grids_sum[i]

            if self.is_crop_enabled:
                grid = np.where(self.static_mask, grid, np.nan)

            overall = self._compute_overall_intensity(grid, mode="hour_sum")

            R = time_slice["R"].isel(time=i).values
            any_link_wet = bool((R > 0).any())
            if any_link_wet:
                overall = max(overall, self.min_if_any)
            overall_str = f"{overall:.3f}"

            png_path = f"{self.outputs_sum_dir}/{fname}_{overall_str}.png"

            rgba = rain_to_rgba_custom(grid, self.hs_levels, self.hs_colors)

            rgba = np.flipud(rgba)
            Image.fromarray(rgba, "RGBA").save(png_path)

            self._write_json_hour_sum(time_slice, i, fname, overall)

        logger.info("[WRITE] Hour-sum PNGs saved.")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _write_json_rain_flags(
        self, calc_dataset: Dataset, t: int, fname: str, overall: float
    ) -> None:
        try:
            os.makedirs(self.outputs_json_dir, exist_ok=True)

            sl = calc_dataset.isel(time=t)
            r = sl.R.values

            true_ids: list[int] = []
            if r is not None and r.size:
                for i in range(sl.cml_id.size):
                    cid = int(sl.cml_id.values[i])
                    v = r[i]
                    if v is None or np.size(v) == 0:
                        any_rain = False
                    else:
                        v = np.asarray(v, dtype=float)
                        any_rain = bool(np.isfinite(v).any() and np.nanmax(v) > 0.0)
                    if any_rain:
                        true_ids.append(cid)

            payload = {
                "utc": fname,
                "overall": float(round(overall, 4)),
                "cml_rain_true": true_ids,
                "count_true": int(len(true_ids)),
                "count_total": int(sl.cml_id.size),
            }

            out_path = os.path.join(self.outputs_json_dir, f"{fname}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.exception(f"[WRITE] JSON output failed for {fname}: {e}")

    def _write_json_hour_sum(
        self, calc_dataset: Dataset, t: int, fname: str, overall: float
    ) -> None:
        try:
            os.makedirs(self.outputs_sum_json_dir, exist_ok=True)
            sl = calc_dataset.isel(time=t)

            if "R_hour_sum" in sl:
                vals = sl["R_hour_sum"].values
            else:
                vals = sl.R.values

            true_ids: list[int] = []
            if vals is not None and np.size(vals):
                for i in range(sl.cml_id.size):
                    cid = int(sl.cml_id.values[i])
                    v = vals[i]
                    if v is None:
                        continue
                    v = np.asarray(v, dtype=float)
                    if v.ndim == 0:
                        any_rain = bool(np.isfinite(v) and v > 0.0)
                    else:
                        finite = np.isfinite(v)
                        any_rain = bool(finite.any() and np.nanmax(v[finite]) > 0.0)
                    if any_rain:
                        true_ids.append(cid)

            payload = {
                "utc": fname,
                "overall": float(round(overall, 4)),
                "cml_rain_true": true_ids,
                "count_true": int(len(true_ids)),
                "count_total": int(sl.cml_id.size),
            }

            out_path = os.path.join(self.outputs_sum_json_dir, f"{fname}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.exception(f"[WRITE] Hour-sum JSON output failed for {fname}: {e}")

    # ------------------------------------------------------------------
    # TIMESERIES WRITING
    # ------------------------------------------------------------------

    def _write_timeseries_intensity_realtime(
        self, calc_dataset: Dataset, np_since_time
    ):
        """Write NORMAL 10-min intensity (mm/h). Controlled by self.skip_influx."""
        filtered = calc_dataset.where(calc_dataset.time > np_since_time).dropna(
            dim="time", how="all"
        )

        points = []
        for c in range(filtered.cml_id.size):
            csl = filtered.isel(cml_id=c)
            cid = int(csl.cml_id)

            rmean = csl.R.mean(dim="channel_id").values

            for t in range(filtered.time.size):
                ts_unix = dt64_to_unixtime(filtered.isel(time=t).time.values)
                points.append(
                    Point("telcorain")
                    .tag("cml_id", cid)
                    .field("rain_intensity", float(rmean[t]))
                    .time(ts_unix, write_precision=WritePrecision.S)
                )

        self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    def _write_timeseries_intensity_historic(self, calc_dataset: Dataset):
        """Write NORMAL 10-min intensity (mm/h). Controlled by self.skip_influx."""
        points = []
        for c in range(calc_dataset.cml_id.size):
            csl = calc_dataset.isel(cml_id=c)
            cid = int(csl.cml_id)

            rmean = csl.R.mean(dim="channel_id").values

            for t in range(calc_dataset.time.size):
                ts_unix = dt64_to_unixtime(calc_dataset.isel(time=t).time.values)
                points.append(
                    Point("telcorain")
                    .tag("cml_id", cid)
                    .field("rain_intensity", float(rmean[t]))
                    .time(ts_unix, write_precision=WritePrecision.S)
                )

        self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    def _write_timeseries_hour_sum_realtime(self, calc_dataset: Dataset, np_since_time):
        """Write hour-sum (mm). Controlled ONLY by config [hour_sum].write_influx."""
        if not bool(self.config.get("hour_sum", {}).get("write_influx", False)):
            return
        if "R_hour_sum" not in calc_dataset.data_vars:
            return

        filtered = calc_dataset.where(calc_dataset.time > np_since_time).dropna(
            dim="time", how="all"
        )
        if "R_hour_sum" not in filtered.data_vars:
            return

        points = []
        for c in range(filtered.cml_id.size):
            csl = filtered.isel(cml_id=c)
            cid = int(csl.cml_id)

            hs = csl["R_hour_sum"]
            if "channel_id" in hs.dims:
                hs = hs.mean(dim="channel_id")
            hs_vals = hs.values

            for t in range(filtered.time.size):
                ts_unix = dt64_to_unixtime(filtered.isel(time=t).time.values)
                v = hs_vals[t]
                if np.isfinite(v):
                    points.append(
                        Point("telcorain")
                        .tag("cml_id", cid)
                        .field("rain_hour_sum", float(v))
                        .time(ts_unix, write_precision=WritePrecision.S)
                    )

        if points:
            self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    def _write_timeseries_hour_sum_historic(self, calc_dataset: Dataset):
        """Write hour-sum (mm). Controlled ONLY by config [hour_sum].write_influx."""
        if not bool(self.config.get("hour_sum", {}).get("write_influx", False)):
            return
        if "R_hour_sum" not in calc_dataset.data_vars:
            return

        points = []
        for c in range(calc_dataset.cml_id.size):
            csl = calc_dataset.isel(cml_id=c)
            cid = int(csl.cml_id)

            hs = csl["R_hour_sum"]
            if "channel_id" in hs.dims:
                hs = hs.mean(dim="channel_id")
            hs_vals = hs.values

            for t in range(calc_dataset.time.size):
                ts_unix = dt64_to_unixtime(calc_dataset.isel(time=t).time.values)
                v = hs_vals[t]
                if np.isfinite(v):
                    points.append(
                        Point("telcorain")
                        .tag("cml_id", cid)
                        .field("rain_hour_sum", float(v))
                        .time(ts_unix, write_precision=WritePrecision.S)
                    )

        if points:
            self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def push_results(
        self, rain_grids, x_grid, y_grid, calc_dataset, rain_grids_sum=None
    ):
        self.influx_man.is_manager_locked = True

        if len(rain_grids) != len(calc_dataset.time):
            logger.error("Raingrids/time mismatch")
            self.influx_man.is_manager_locked = False
            return

        # Historic warmup trimming
        hist_cfg = self.config.get("historic", {})
        if self.is_historic and hist_cfg.get("compensate_historic", False):
            req_start = self.config["time"]["start"]

            if req_start.tzinfo:
                req_start_utc = req_start.astimezone(timezone.utc)
            else:
                req_start_utc = req_start.replace(tzinfo=timezone.utc)

            req_start_utc = req_start_utc.replace(tzinfo=None)

            calc_dataset = calc_dataset.sel(time=slice(req_start_utc, None))
            keep_len = calc_dataset.sizes["time"]
            rain_grids = rain_grids[-keep_len:]
            if rain_grids_sum is not None:
                rain_grids_sum = rain_grids_sum[-keep_len:]

        if rain_grids_sum is not None and self.config["hour_sum"]["enabled"]:
            if len(rain_grids_sum) != len(calc_dataset.time):
                m = min(len(rain_grids_sum), len(calc_dataset.time))
                logger.warning(
                    f"[WRITE] Aligning hour-sum to min length {m}: "
                    f"grids_sum={len(rain_grids_sum)} time={len(calc_dataset.time)}"
                )
                rain_grids_sum = rain_grids_sum[-m:]
                calc_dataset = calc_dataset.isel(time=slice(-m, None))
                rain_grids = rain_grids[-m:]

        # ensure dirs
        if self.config["directories"]["save_web"]:
            os.makedirs(self.outputs_web_dir, exist_ok=True)
        if self.config["directories"]["save_raw"]:
            os.makedirs(self.outputs_raw_dir, exist_ok=True)
        if self.output_json_info:
            os.makedirs(self.outputs_json_dir, exist_ok=True)

        # write PNGs
        self._write_raingrids(rain_grids, x_grid, y_grid, calc_dataset)

        if self.config["hour_sum"]["enabled"] and rain_grids_sum is not None:
            self._write_sum_raingrids(rain_grids_sum, x_grid, y_grid, calc_dataset)

        # write timeseries
        if self.is_historic:
            # normal intensity only if skip_influx is False
            if not self.skip_influx:
                self._write_timeseries_intensity_historic(calc_dataset)
            # hour-sum independently from skip_influx
            self._write_timeseries_hour_sum_historic(calc_dataset)
        else:
            np_since = np.datetime64(self.since_time)
            # normal intensity only if skip_influx is False
            if not self.skip_influx:
                self._write_timeseries_intensity_realtime(calc_dataset, np_since)
            # hour-sum independently from skip_influx
            self._write_timeseries_hour_sum_realtime(calc_dataset, np_since)

        self.influx_man.is_manager_locked = False
