import json
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from shapely.geometry import Point as GeoPoint
from shapely.geometry import shape
from shapely.prepared import PreparedGeometry, prep
from xarray import Dataset

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import sql_man
from telcorain.handlers import config_handler
from telcorain.handlers.logging_handler import logger
from telcorain.handlers.writer import mask_grid, ndarray_to_png, save_ndarray_to_file
from telcorain.procedures.calculation import Calculation
from telcorain.procedures.utils.helpers import dt64_to_unixtime


class HistoricWriter:

    def __init__(self, cp: dict, link_ids: list[int] = None):
        self.cp = cp
        self.sql_man = sql_man
        self.influx_man = influx_man
        self.links = sql_man.load_metadata()
        self.selected_links = None
        if not link_ids:
            self._select_all_links()
        else:
            self._select_links(link_ids=link_ids)
        self.calculation = Calculation(
            influx_man=influx_man,
            results_id=0,
            links=self.links,
            selection=self.selected_links,
            cp=self.cp,
        )

        self.is_crop_enabled = config_handler.read_option(
            "realtime", "crop_to_geojson_polygon"
        )
        self.geojson_file = config_handler.read_option("realtime", "geojson")
        self.output_dir = config_handler.read_option("directories", "outputs_web")
        self.outputs_raw_dir = config_handler.read_option("directories", "outputs_raw")

    def _select_all_links(self) -> None:
        selected_links = {}
        for link in self.links:
            selected_links[self.links[link].link_id] = 3
        self.selected_links = selected_links

    def _select_links(self, link_ids) -> None:
        selected_links = {}
        for link_id in link_ids:
            selected_links[link_id] = 3
        self.selected_links = selected_links

    def _write_raingrids(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
        np_last_time: np.datetime64 = None,
        np_since_time: np.datetime64 = None,
    ):
        """
        Write raingrids metadata into MariaDB table and save them as PNG images and NPY raw data (if enabled).
        :param rain_grids: list of 2D numpy arrays with rain intensity values
        :param x_grid: 2D numpy array of x coordinates
        :param y_grid: 2D numpy array of y coordinates
        :param calc_dataset: xarray Dataset with calculation data
        :param np_last_time: last raingrid time in the database
        :param np_since_time: time since last realtime calculation start (overwritten by historic write)
        """
        prepared_polygons: Optional[list[PreparedGeometry]] = None
        if self.is_crop_enabled:
            with open(f"./assets/{self.geojson_file}") as f:
                geojson = json.load(f)
            polygons = [shape(feature["geometry"]) for feature in geojson["features"]]
            prepared_polygons = [prep(polygon) for polygon in polygons]
            logger.debug(
                '[WRITE] GeoJSON file "%s" loaded. %d polygons found.',
                self.geojson_file,
                len(polygons),
            )

        for t in range(len(calc_dataset.time)):
            time = calc_dataset.time[t]
            # if (time.values > np_last_time) and (
            #     self.write_historic or (time.values > np_since_time)
            # ):
            raingrid_time: datetime = datetime.utcfromtimestamp(
                dt64_to_unixtime(time.values)
            )
            formatted_time: str = raingrid_time.strftime("%Y-%m-%d %H:%M")
            file_name: str = raingrid_time.strftime("%Y-%m-%d_%H%M")

            logger.info("[WRITE] Saving raingrid %s for web output...", formatted_time)
            raingrid_links = calc_dataset.isel(time=t).cml_id.values.tolist()
            # get avg/max rain intensity value
            r_median_value = np.nanmedian(rain_grids[t])
            r_avg_value = np.nanmean(rain_grids[t])
            r_max_value = np.nanmax(rain_grids[t])
            logger.debug(
                "[WRITE] Writing raingrid's %s metadata into MariaDB...",
                formatted_time,
            )

            # self.sql_man.insert_raingrid(
            #     time=raingrid_time,
            #     links=raingrid_links,
            #     file_name=f"{file_name}.png",
            #     r_median=r_median_value,
            #     r_avg=r_avg_value,
            #     r_max=r_max_value,
            # )

            rain_grid = rain_grids[t]
            if self.is_crop_enabled:
                logger.debug(
                    "[WRITE] Cropping raingrid %s to the GeoJSON polygon(s)...",
                    formatted_time,
                )
                rain_grid = mask_grid(rain_grid, x_grid, y_grid, prepared_polygons)

            logger.debug("[WRITE] Saving raingrid %s as PNG file...", formatted_time)
            ndarray_to_png(rain_grid, f"{self.output_dir}/{file_name}.png")

            logger.debug(
                "[WRITE] Saving raingrid %s as raw numpy file...", formatted_time
            )
            save_ndarray_to_file(rain_grid, f"{self.outputs_raw_dir}/{file_name}.npy")

            logger.debug("[WRITE] Raingrid %s successfully saved.", formatted_time)

        logger.info("[WRITE] Saving raingrids – DONE.")

    def write_raingrids(self) -> None:
        x_coords = np.arange(
            self.cp["X_MIN"], self.cp["X_MAX"], self.cp["interpol_res"]
        )
        y_coords = np.arange(
            self.cp["Y_MIN"], self.cp["Y_MAX"], self.cp["interpol_res"]
        )
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        self.calculation.run()
        self._write_raingrids(
            rain_grids=self.calculation.rain_grids,
            x_grid=x_grid,
            y_grid=y_grid,
            calc_dataset=self.calculation.calc_data,
        )
