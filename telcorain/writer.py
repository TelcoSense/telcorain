import json
import os
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
        if is_historic:
            user_dir = self.config["user_info"]["folder_name"]
            self.outputs_raw_dir = f"outputs_historic/{user_dir}_raw"
            self.outputs_web_dir = f"outputs_historic/{user_dir}_web"
        else:
            self.outputs_raw_dir = self.config["directories"]["outputs_raw"]
            self.outputs_web_dir = self.config["directories"]["outputs_web"]

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
            png_path = f"{self.outputs_web_dir}/{fname}.png"

            need_raw = self.config["directories"]["save_raw"] and not os.path.exists(
                raw_path
            )
            need_png = self.config["directories"]["save_web"] and not os.path.exists(
                png_path
            )

            if not (need_raw or need_png):
                continue

            grid = rain_grids[t]

            # crop polygon
            if self.is_crop_enabled:
                grid = np.where(self.static_mask, grid, np.nan)

            # Save raw
            if need_raw:
                save_ndarray_to_file(grid, raw_path)

            # Save PNG
            if need_png:
                rgba = rain_to_rgba(grid)
                rgba = np.flipud(rgba)
                Image.fromarray(rgba, "RGBA").save(png_path)

        logger.info("[WRITE] PNGs saved.")

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
