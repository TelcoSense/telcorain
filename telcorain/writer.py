"""Module containing the Writer class for writing results of the calculation."""

import json
import os
from datetime import datetime, timezone
from threading import Thread
from typing import Optional
from PIL import Image

import numpy as np
from influxdb_client import Point, WritePrecision
from xarray import Dataset
from shapely.geometry import shape, Point as GeoPoint
from shapely.prepared import prep
from shapely.ops import unary_union

from telcorain.helpers import dt64_to_unixtime, save_ndarray_to_file
from telcorain.cython.raincolor import rain_to_rgba
from telcorain.handlers import logger


class Writer:

    def __init__(
        self,
        sql_man,
        influx_man,
        skip_influx: bool,
        skip_sql: bool,
        config: dict,
        since_time: Optional[datetime] = None,
        is_historic: bool = False,
        influx_wipe_thread: Optional[Thread] = None,
    ):
        self.sql_man = sql_man
        self.influx_man = influx_man
        self.skip_influx = skip_influx
        self.skip_sql = skip_sql
        self.config = config
        self.is_historic = is_historic
        self.influx_wipe_thread = influx_wipe_thread

        self.is_crop_enabled = self.config["rendering"]["is_crop_enabled"]
        self.geojson_file = self.config["rendering"]["geojson_file"]

        if self.is_historic:
            user_dir = self.config["user_info"]["folder_name"]
            self.outputs_raw_dir = f"outputs_historic/{user_dir}_raw"
            self.outputs_web_dir = f"outputs_historic/{user_dir}_web"
        else:
            self.outputs_raw_dir = self.config["directories"]["outputs_raw"]
            self.outputs_web_dir = self.config["directories"]["outputs_web"]

        if since_time is None:
            since_time = datetime.min

        self.since_time = since_time

        # STATIC MASK CACHE
        self.static_mask = None
        self.prep_poly = None
        self.bbox = None

    # ------------------------------------------------------------------
    # POLYGON → STATIC MASK
    # ------------------------------------------------------------------

    def _load_polygon(self):
        """Load and merge GeoJSON polygons."""
        with open(f"./assets/{self.geojson_file}", "r", encoding="utf-8") as f:
            data = json.load(f)

        polys = [shape(f["geometry"]).buffer(0) for f in data["features"]]
        merged = unary_union(polys).buffer(0)
        prep_poly = prep(merged)

        bbox = merged.bounds  # (minx, miny, maxx, maxy)
        return prep_poly, bbox

    def _compute_static_mask(self, x_grid, y_grid):
        """
        Precompute polygon boolean mask once.
        Extremely fast during writing.
        """

        minx, miny, maxx, maxy = self.bbox

        bbox_mask = (
            (x_grid >= minx) & (x_grid <= maxx) & (y_grid >= miny) & (y_grid <= maxy)
        )

        static_mask = np.zeros_like(bbox_mask, dtype=bool)

        pts = np.column_stack((x_grid[bbox_mask], y_grid[bbox_mask]))

        flat_mask = []
        for lon, lat in pts:
            flat_mask.append(self.prep_poly.contains(GeoPoint(lon, lat)))

        static_mask[bbox_mask] = flat_mask
        return static_mask

    # ------------------------------------------------------------------
    # RAINGRID WRITER
    # ------------------------------------------------------------------

    def _write_raingrids(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ):

        # ------------------------------------------------------
        # Prepare polygon and static mask ONCE
        # ------------------------------------------------------
        if self.is_crop_enabled and self.static_mask is None:
            logger.debug("[WRITE] Building static polygon mask...")

            self.prep_poly, self.bbox = self._load_polygon()
            self.static_mask = self._compute_static_mask(x_grid, y_grid)

            logger.debug(
                "[WRITE] Static mask computed: %s pixels inside polygon",
                np.sum(self.static_mask),
            )

        for t in range(len(calc_dataset.time)):

            # ---------- timestamps ----------
            time_val = calc_dataset.time[t]
            raingrid_time = datetime.utcfromtimestamp(dt64_to_unixtime(time_val.values))
            file_name = raingrid_time.strftime("%Y-%m-%d_%H%M")
            raw_path = f"{self.outputs_raw_dir}/{file_name}.npy"
            png_path = f"{self.outputs_web_dir}/{file_name}.png"

            need_raw = self.config["directories"]["save_raw"] and not os.path.exists(
                raw_path
            )
            need_png = self.config["directories"]["save_web"] and not os.path.exists(
                png_path
            )

            if not need_raw and not need_png:
                continue

            # ---------- select grid ----------
            grid = rain_grids[t]

            # ---------- apply mask ----------
            if self.is_crop_enabled:
                grid = np.where(self.static_mask, grid, np.nan)

            # ---------- RAW ----------
            if need_raw:
                if not self.skip_sql:
                    links = calc_dataset.isel(time=t).cml_id.values.tolist()
                    r_median = np.nanmedian(grid)
                    r_avg = np.nanmean(grid)
                    r_max = np.nanmax(grid)

                    self.sql_man.insert_raingrid(
                        time=raingrid_time,
                        links=links,
                        file_name=f"{file_name}.png",
                        r_median=r_median,
                        r_avg=r_avg,
                        r_max=r_max,
                    )

                save_ndarray_to_file(grid, raw_path)

            # ---------- PNG ----------
            if need_png:
                rgba = rain_to_rgba(grid)
                rgba = np.flipud(rgba)
                Image.fromarray(rgba, "RGBA").save(png_path)

        logger.info("[WRITE] Saving raingrids locally -- DONE.")

    # ------------------------------------------------------------------
    # TIMESERIES
    # ------------------------------------------------------------------

    def _write_timeseries_realtime(self, calc_dataset, np_last_time, np_since_time):

        compare_time = np_since_time if np_since_time > np_last_time else np_last_time
        logger.debug("[WRITE: InfluxDB] Preparing realtime rain timeseries...")

        filtered = calc_dataset.where(calc_dataset.time > compare_time).dropna(
            dim="time", how="all"
        )

        cmls_count = filtered.cml_id.size
        times_count = filtered.time.size

        points_to_write = []

        if cmls_count > 0 and times_count > 0:
            for cml in range(cmls_count):
                for t in range(times_count):
                    points_to_write.append(
                        Point("telcorain")
                        .tag("cml_id", int(filtered.isel(cml_id=cml).cml_id))
                        .field(
                            "rain_intensity",
                            float(
                                filtered.isel(cml_id=cml)
                                .R.mean(dim="channel_id")
                                .isel(time=t)
                            ),
                        )
                        .time(
                            dt64_to_unixtime(filtered.isel(time=t).time.values),
                            write_precision=WritePrecision.S,
                        )
                    )

        self.influx_man.write_points(points_to_write, self.influx_man.BUCKET_OUT_CML)

    def _write_timeseries_historic(self, calc_dataset):

        logger.info("[WRITE: InfluxDB] Preparing historic rain timeseries...")
        points_to_write = []

        for cml in range(calc_dataset.cml_id.size):
            for t in range(calc_dataset.time.size):
                points_to_write.append(
                    Point("telcorain")
                    .tag("cml_id", int(calc_dataset.isel(cml_id=cml).cml_id))
                    .field(
                        "rain_intensity",
                        float(
                            calc_dataset.isel(cml_id=cml)
                            .R.mean(dim="channel_id")
                            .isel(time=t)
                        ),
                    )
                    .time(
                        dt64_to_unixtime(calc_dataset.isel(time=t).time.values),
                        write_precision=WritePrecision.S,
                    )
                )

        self.influx_man.write_points(points_to_write, self.influx_man.BUCKET_OUT_CML)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def push_results(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ):
        self.influx_man.is_manager_locked = True

        if len(rain_grids) != len(calc_dataset.time):
            logger.error("Raingrids/time mismatch")
            self.influx_man.is_manager_locked = False
            return

        # Historic compensation
        if self.is_historic and self.config.get("historic", {}).get(
            "compensate_historic", False
        ):
            desired_start = self.config["time"]["start"]
            desired_start = desired_start.astimezone(timezone.utc).replace(tzinfo=None)
            calc_dataset = calc_dataset.sel(time=slice(desired_start, None))
            time_len = calc_dataset.sizes["time"]
            rain_grids = rain_grids[-time_len:]

        # Ensure directories
        if self.config["directories"]["save_web"]:
            os.makedirs(self.outputs_web_dir, exist_ok=True)
        if self.config["directories"]["save_raw"]:
            os.makedirs(self.outputs_raw_dir, exist_ok=True)

        # I. raingrids
        self._write_raingrids(rain_grids, x_grid, y_grid, calc_dataset)

        # II. timeseries
        if not self.skip_influx:
            if self.is_historic:
                self._write_timeseries_historic(calc_dataset)
            else:
                last_record = self.sql_man.get_last_raingrid()
                last_time = list(last_record.keys())[0] if last_record else datetime.min
                np_last_time = np.datetime64(last_time)
                np_since_time = np.datetime64(self.since_time)
                self._write_timeseries_realtime(
                    calc_dataset, np_last_time, np_since_time
                )

        self.influx_man.is_manager_locked = False
