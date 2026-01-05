import json
import os
import glob
import math
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

import numpy as np
from PIL import Image
from influxdb_client import Point, WritePrecision
from xarray import Dataset
from shapely.geometry import shape, Point as GeoPoint
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep
from pyproj import Transformer

from telcorain.handlers import logger
from telcorain.helpers import dt64_to_unixtime, save_ndarray_to_file
from telcorain.cython.raincolor import rain_to_rgba


class Writer:
    def __init__(
        self,
        influx_man,
        skip_influx: bool,
        config: dict,
        since_time: Optional[datetime] = None,
        is_historic: bool = False,
        influx_wipe_thread: Optional[Thread] = None,
    ):
        self.influx_man = influx_man
        self.skip_influx = skip_influx
        self.config = config
        self.is_historic = is_historic
        self.influx_wipe_thread = influx_wipe_thread

        self.is_crop_enabled = self.config["rendering"]["is_crop_enabled"]
        self.geojson_file = self.config["rendering"]["geojson_file"]

        # output dirs
        if self.is_historic:
            user_dir = self.config["user_info"]["folder_name"]
            self.outputs_raw_dir = f"outputs_historic/{user_dir}_raw"
            self.outputs_web_dir = f"outputs_historic/{user_dir}_web"
        else:
            self.outputs_raw_dir = self.config["directories"]["outputs_raw"]
            self.outputs_web_dir = self.config["directories"]["outputs_web"]

        # Optional JSON output (realtime-only)
        self.outputs_json_dir = self.config.get("directories", {}).get(
            "outputs_json", "./outputs_json"
        )
        self.output_json_info = bool(
            self.config.get("realtime", {}).get("output_json_info", False)
        )

        # Overall rain intensity (0..1) for per-frame naming
        rg_cfg = self.config.get("raingrids", {})
        self.overall_intensity_ref = float(rg_cfg.get("overall_intensity_ref", 20.0))
        self.overall_intensity_method = str(
            rg_cfg.get("overall_intensity_method", "mean")
        ).lower()
        self.overall_intensity_threshold = float(
            rg_cfg.get("overall_intensity_threshold", rg_cfg.get("min_rain_value", 0))
        )
        self.overall_intensity_coverage_gamma = float(
            rg_cfg.get("overall_intensity_coverage_gamma", 0.3)
        )
        self.overall_intensity_strength_gamma = float(
            rg_cfg.get("overall_intensity_strength_gamma", 1.0)
        )
        self.min_if_any = float(rg_cfg.get("overall_intensity_min_if_any_link", 0.001))

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
            # Transformation to EPSG:3857 → EPSG:4326 bounds for PNG overlay
            self.transform_back = Transformer.from_crs(
                "EPSG:3857", "EPSG:4326", always_xy=True
            )
        else:
            self.transform_fwd = None
            self.transform_back = None

    # ------------------------------------------------------------------
    # POLYGON MASKING
    # ------------------------------------------------------------------

    def _load_polygon(self):
        """Load GeoJSON mask, reproject to Mercator if needed."""
        with open(f"./assets/{self.geojson_file}", "r", encoding="utf-8") as f:
            data = json.load(f)

        polys = [shape(f["geometry"]).buffer(0) for f in data["features"]]

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

    def _compute_overall_intensity(self, grid: np.ndarray) -> float:
        """Compute a single 0..1 intensity value from a rain grid (mm/h).

        Designed to be more sensitive near 0 using log1p scaling.
        """
        if grid is None:
            return 0.0
        # Ignore NaNs (outside mask / missing)
        valid = np.asarray(grid, dtype=float)
        if valid.size == 0:
            return 0.0

        method = self.overall_intensity_method

        # Coverage-based methods
        if method in {"coverage", "coverage_strength"}:
            finite = np.isfinite(valid)
            denom = int(np.count_nonzero(finite))
            if denom == 0:
                return 0.0
            wet = finite & (valid > self.overall_intensity_threshold)
            coverage = float(np.count_nonzero(wet)) / float(denom)
            if method == "coverage":
                return float(min(1.0, max(0.0, coverage)))

            # Strength computed only on wet pixels (above threshold)
            wet_vals = valid[wet]
            if wet_vals.size == 0:
                return 0.0
            stat = float(np.nanmean(wet_vals))
            if not math.isfinite(stat) or stat <= 0:
                return 0.0

            ref = max(self.overall_intensity_ref, 1e-6)
            strength = math.log1p(stat) / math.log1p(ref)
            strength = float(min(1.0, max(0.0, strength)))

            # Nonlinear combination: coverage drives "how much of Czechia is wet",
            # strength drives "how hard it rains".
            cg = max(self.overall_intensity_coverage_gamma, 1e-6)
            sg = max(self.overall_intensity_strength_gamma, 1e-6)
            score = (coverage**cg) * (strength**sg)
            return float(min(1.0, max(0.0, score)))

        # Value-statistic methods (legacy)
        if method == "p90":
            stat = float(np.nanpercentile(valid, 90))
        elif method == "p95":
            stat = float(np.nanpercentile(valid, 95))
        else:
            stat = float(np.nanmean(valid))

        if not math.isfinite(stat) or stat <= 0:
            return 0.0

        ref = max(self.overall_intensity_ref, 1e-6)
        # log scaling gives better separation near 0
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

        # Log WGS84 bounds (for OSM overlay)
        if self.use_mercator:
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

            # PNG name includes overall intensity: <UTC>_<0..1>.png
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

            # crop polygon
            if self.is_crop_enabled:
                grid = np.where(self.static_mask, grid, np.nan)

            # compute overall intensity (0..1) on the cropped grid
            overall = self._compute_overall_intensity(grid)

            # if any link is wet, than overall is at least self.min_if_any
            R = calc_dataset["R"].isel(time=t).values
            any_link_wet = bool((R > 0).any())

            if any_link_wet:
                overall = max(overall, self.min_if_any)
            overall_str = f"{overall:.3f}"

            png_path = f"{self.outputs_web_dir}/{fname}_{overall_str}.png"

            # Save raw
            if need_raw:
                save_ndarray_to_file(grid, raw_path)

            # Save PNG
            if need_png:
                rgba = rain_to_rgba(grid)
                rgba = np.flipud(rgba)
                Image.fromarray(rgba, "RGBA").save(png_path)

            # Optional JSON per-frame summary (realtime-only)
            if not self.is_historic and self.output_json_info:
                self._write_json_rain_flags(calc_dataset, t, fname, overall)

        logger.info("[WRITE] PNGs saved.")

    # ------------------------------------------------------------------
    # JSON WRITER (realtime-only)
    # ------------------------------------------------------------------

    def _write_json_rain_flags(
        self, calc_dataset: Dataset, t: int, fname: str, overall: float
    ) -> None:
        """Write per-frame JSON with only CML IDs where rain is present.

        Output format:
            {
            "utc": "YYYY-MM-DD_HHMM",
            "overall": 0.021,
            "cml_rain_true": [123, 456, 789],
            "count_true": 3,
            "count_total": 331
            }
        """
        try:
            os.makedirs(self.outputs_json_dir, exist_ok=True)

            sl = calc_dataset.isel(time=t)
            r = sl.R.values

            true_ids: list[int] = []
            if r is not None and r.size:
                # any_rain if any channel > 0 (ignore NaNs)
                # r shape: (n_cml, n_ch)
                for i in range(sl.cml_id.size):
                    cid = int(sl.cml_id.values[i])
                    v = r[i]
                    any_rain = bool(np.nanmax(v) > 0.0) if np.size(v) else False
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

    # ------------------------------------------------------------------
    # TIMESERIES WRITING
    # ------------------------------------------------------------------

    def _write_timeseries_realtime(self, calc_dataset: Dataset, np_since_time):
        filtered = calc_dataset.where(calc_dataset.time > np_since_time).dropna(
            dim="time", how="all"
        )
        points = []
        for c in range(filtered.cml_id.size):
            csl = filtered.isel(cml_id=c)
            cid = int(csl.cml_id)
            rmean = csl.R.mean(dim="channel_id").values
            for t in range(filtered.time.size):
                points.append(
                    Point("telcorain")
                    .tag("cml_id", cid)
                    .field("rain_intensity", float(rmean[t]))
                    .time(
                        dt64_to_unixtime(filtered.isel(time=t).time.values),
                        write_precision=WritePrecision.S,
                    )
                )
        self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    def _write_timeseries_historic(self, calc_dataset: Dataset):
        points = []
        for c in range(calc_dataset.cml_id.size):
            csl = calc_dataset.isel(cml_id=c)
            cid = int(csl.cml_id)
            rmean = csl.R.mean(dim="channel_id").values
            for t in range(calc_dataset.time.size):
                points.append(
                    Point("telcorain")
                    .tag("cml_id", cid)
                    .field("rain_intensity", float(rmean[t]))
                    .time(
                        dt64_to_unixtime(calc_dataset.isel(time=t).time.values),
                        write_precision=WritePrecision.S,
                    )
                )
        self.influx_man.write_points(points, self.influx_man.BUCKET_OUT_CML)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def push_results(self, rain_grids, x_grid, y_grid, calc_dataset):
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

        # ensure dirs
        if self.config["directories"]["save_web"]:
            os.makedirs(self.outputs_web_dir, exist_ok=True)
        if self.config["directories"]["save_raw"]:
            os.makedirs(self.outputs_raw_dir, exist_ok=True)
        if not self.is_historic and self.output_json_info:
            os.makedirs(self.outputs_json_dir, exist_ok=True)

        # write PNGs
        self._write_raingrids(rain_grids, x_grid, y_grid, calc_dataset)

        # write timeseries
        if not self.skip_influx:
            if self.is_historic:
                self._write_timeseries_historic(calc_dataset)
            else:
                np_since = np.datetime64(self.since_time)
                self._write_timeseries_realtime(calc_dataset, np_since)

        self.influx_man.is_manager_locked = False
