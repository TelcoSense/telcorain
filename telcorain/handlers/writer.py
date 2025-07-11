"""Module containing the Writer class for writing results of the calculation."""

import json
import os
from datetime import datetime
from threading import Thread
from typing import Optional

import numpy as np
from influxdb_client import Point, WritePrecision
from PIL import Image
from shapely.geometry import Point as GeoPoint
from shapely.geometry import shape
from shapely.prepared import PreparedGeometry, prep
from xarray import Dataset

from telcorain.database.influx_manager import InfluxManager
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers import config_handler
from telcorain.handlers.logging_handler import logger
from telcorain.procedures.utils.helpers import dt64_to_unixtime


class RealtimeWriter:
    """
    Class for writing results of the real-time calculation into the database and outputs directory.
    Raingrid metadata are written into MariaDB.
    The raingrids themselves are saved as PNG images in the outputs directory, from where they are then served by
    Telcorain's HTTP server (if enabled) to the web application (or other clients).
    Also, raw numpy NPY files are saved in the outputs_raw directory, which are then processed by the HTTP REST API
    endpoint for the queries on the exact rain intensity values on the given coordinates.
    Then, individual CML timeseries are written into InfluxDB.

    The class is designed to be run primarily in a separate thread to prevent blocking of the GUI thread.
    """

    def __init__(
        self,
        sql_man: SqlManager,
        influx_man: InfluxManager,
        skip_influx: bool,
        skip_sql: bool,
        since_time: datetime,
        cp: dict,
        config: dict,
        influx_wipe_thread: Optional[Thread] = None,
    ):
        """
        Initialize the RealtimeWriter object. The object is used to write results of the real-time calculation.
        :param sql_man: SQL manager object
        :param influx_man: InfluxDB manager object
        :param skip_influx: flag for skipping InfluxDB timeseries writing
        :param since_time: time since last realtime calculation start (overwritten by historic write)
        :param influx_wipe_thread: a thread with InfluxDB wiping activity for checking if it is done (if forced write)
        """
        self.sql_man = sql_man
        self.influx_man = influx_man
        self.skip_influx = skip_influx
        self.skip_sql = skip_sql
        self.since_time = since_time
        self.influx_wipe_thread = influx_wipe_thread
        self.cp = cp
        self.config = config
        self.is_crop_enabled = self.cp["rendering"]["is_crop_enabled"]
        self.geojson_file = self.cp["rendering"]["geojson_file"]
        self.output_dir = self.config["directories"]["outputs_web"]
        self.outputs_raw_dir = self.config["directories"]["outputs_raw"]

    def _write_raingrids(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
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
            raingrid_time: datetime = datetime.utcfromtimestamp(
                dt64_to_unixtime(time.values)
            )
            formatted_time: str = raingrid_time.strftime("%Y-%m-%d %H:%M")
            file_name: str = raingrid_time.strftime("%Y-%m-%d_%H%M")

            if not os.path.exists(f"{self.outputs_raw_dir}/{file_name}.npy"):
                logger.debug(
                    "[WRITE] Saving raingrid %s for web output...", formatted_time
                )
                raingrid_links = calc_dataset.isel(time=t).cml_id.values.tolist()
                # get avg/max rain intensity value
                r_median_value = np.nanmedian(rain_grids[t])
                r_avg_value = np.nanmean(rain_grids[t])
                r_max_value = np.nanmax(rain_grids[t])

                if not self.skip_sql:
                    logger.debug(
                        "[WRITE] Writing raingrid's %s metadata into MariaDB...",
                        formatted_time,
                    )
                    self.sql_man.insert_raingrid(
                        time=raingrid_time,
                        links=raingrid_links,
                        file_name=f"{file_name}.png",
                        r_median=r_median_value,
                        r_avg=r_avg_value,
                        r_max=r_max_value,
                    )

                rain_grid = rain_grids[t]

                if self.is_crop_enabled:
                    logger.debug(
                        "[WRITE] Cropping raingrid %s to the GeoJSON polygon(s)...",
                        formatted_time,
                    )
                    rain_grid = mask_grid(rain_grid, x_grid, y_grid, prepared_polygons)

                logger.debug(
                    "[WRITE] Saving raingrid %s as PNG file...", formatted_time
                )
                ndarray_to_png(rain_grid, f"{self.output_dir}/{file_name}.png")

                logger.debug(
                    "[WRITE] Saving raingrid %s as raw numpy file...", formatted_time
                )
                save_ndarray_to_file(
                    rain_grid, f"{self.outputs_raw_dir}/{file_name}.npy"
                )

                logger.debug("[WRITE] Raingrid %s successfully saved.", formatted_time)

        logger.info("[WRITE] Saving raingrids locally -- DONE.")

    def _write_timeseries(
        self,
        calc_dataset: Dataset,
        np_last_time: np.datetime64,
        np_since_time: np.datetime64,
    ):
        """
        Write individual CML rain instensity timeseries into InfluxDB.
        :param calc_dataset: xarray Dataset with calculation data, containing CML timeseries with rain intensity
        :param np_last_time: last raingrid time in the database
        """
        if np_since_time > np_last_time:
            compare_time = np_since_time
        else:
            compare_time = np_last_time

        points_to_write = []

        logger.debug(
            "[WRITE: InfluxDB] Preparing rain timeseries from individual CMLs for writing into InfluxDB..."
        )

        filtered = calc_dataset.where(calc_dataset.time > compare_time).dropna(
            dim="time", how="all"
        )
        cmls_count = filtered.cml_id.size
        times_count = filtered.time.size

        if (cmls_count > 0) and (times_count > 0):
            for cml in range(cmls_count):
                for time in range(times_count):
                    points_to_write.append(
                        Point("telcorain")
                        .tag("cml_id", int(filtered.isel(cml_id=cml).cml_id))
                        .field(
                            "rain_intensity",
                            float(
                                filtered.isel(cml_id=cml)
                                .R.mean(dim="channel_id")
                                .isel(time=time)
                            ),
                        )
                        .time(
                            dt64_to_unixtime(filtered.isel(time=time).time.values),
                            write_precision=WritePrecision.S,
                        )
                    )

        if self.influx_wipe_thread is not None:
            logger.debug(
                "[WRITE: InfluxDB] Force write is active. Checking if InfluxDB wipe thread is done..."
            )
            self.influx_wipe_thread.join()
            logger.debug(
                "[WRITE: InfluxDB] Wipe thread is done. Proceeding with timeseries writing..."
            )
        logger.debug(
            "[WRITE: InfluxDB] Writing rain timeseries from individual CMLs into database..."
        )
        self.influx_man.write_points(points_to_write, self.influx_man.BUCKET_OUT_CML)
        logger.debug(
            "[WRITE: InfluxDB] Writing rain timeseries from individual CMLs -- DONE."
        )

    def push_results(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ):
        """
        Push the results of the real-time calculation into the database and outputs directory.
        :param rain_grids: list of 2D numpy arrays with rain intensity values
        :param x_grid: 2D numpy array of x coordinates
        :param y_grid: 2D numpy array of y coordinates
        :param calc_dataset: xarray Dataset with calculation data
        """
        # lock the influx manager to prevent start of new calculation before writing is finished
        # (lock is checked by the calculation starting mechanism, if the lock is active, calculation cannot start)
        self.influx_man.is_manager_locked = True

        if len(rain_grids) != len(calc_dataset.time):
            logger.error(
                "Cannot write raingrids into DB! Inconsistent count of rain grid frames "
                "(%d) and times in calculation dataset (%d)!",
                len(rain_grids),
                len(calc_dataset.time),
            )
            return

        last_record = self.sql_man.get_last_raingrid()

        if len(last_record) > 0:
            last_time = list(last_record.keys())[0]
        else:
            last_time = datetime.min

        np_last_time = np.datetime64(last_time)
        np_since_time = np.datetime64(self.since_time)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.outputs_raw_dir, exist_ok=True)

        # I. RAINGRIDS INTO MARIADB
        self._write_raingrids(
            rain_grids,
            x_grid,
            y_grid,
            calc_dataset,
        )
        del rain_grids

        # II. INDIVIDUAL CML TIMESERIES INTO INFLUXDB
        if not self.skip_influx:
            self._write_timeseries(calc_dataset, np_last_time, np_since_time)
        del calc_dataset

        # unlock the influx manager
        self.influx_man.is_manager_locked = False

    def start_push_results_thread(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ) -> Thread:
        """
        Start a thread for pushing the results of the real-time calculation into the database and outputs directory.
        :param rain_grids: list of 2D numpy arrays with rain intensity values
        :param x_grid: 2D numpy array of x coordinates
        :param y_grid: 2D numpy array of y coordinates
        :param calc_dataset: xarray Dataset with calculation data
        :return: the thread object with the started activity
        """
        thread = Thread(
            target=self.push_results, args=(rain_grids, x_grid, y_grid, calc_dataset)
        )
        thread.start()
        return thread


class Writer:
    """
    Class for writing results of the calculation into the database and outputs directory.
    Raingrid metadata can be written into MariaDB.
    The raingrids themselves are saved as PNG images in the outputs directory, from where they are then served by
    Telcorain's HTTP server (if enabled) to the web application (or other clients).
    Also, raw numpy NPY files are saved in the outputs_raw directory, which are then processed by the HTTP REST API
    endpoint for the queries on the exact rain intensity values on the given coordinates.
    Then, individual CML timeseries are written into InfluxDB.

    The class is designed to be run primarily in a separate thread to prevent blocking of the GUI thread.
    """

    def __init__(
        self,
        sql_man: SqlManager,
        influx_man: InfluxManager,
        write_historic: bool,
        skip_influx: bool,
        skip_sql: bool,
        cp: dict,
        config: dict,
        influx_wipe_thread: Optional[Thread] = None,
    ):
        """
        Initialize the RealtimeWriter object. The object is used to write results of the real-time calculation.
        :param sql_man: SQL manager object
        :param influx_man: InfluxDB manager object
        :param write_historic: flag for writing historic results, overwriting the since_time parameter
        :param skip_influx: flag for skipping InfluxDB timeseries writing
        :param since_time: time since last realtime calculation start (overwritten by historic write)
        :param cp: calculation params dictionary
        :param influx_wipe_thread: a thread with InfluxDB wiping activity for checking if it is done (if forced write)
        """
        self.sql_man = sql_man
        self.influx_man = influx_man
        self.write_historic = write_historic
        self.skip_influx = skip_influx
        self.skip_sql = skip_sql
        self.influx_wipe_thread = influx_wipe_thread
        self.cp = cp
        self.config = config
        self.is_crop_enabled = self.cp["rendering"]["is_crop_enabled"]
        self.geojson_file = self.cp["rendering"]["geojson_file"]
        self.user_dir = self.cp["user_info"]["folder_name"]
        self.output_dir = f"outputs_historic/{self.user_dir}_web"
        self.outputs_raw_dir = f"outputs_historic/{self.user_dir}_raw"

    def _write_raingrids(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
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
            raingrid_time: datetime = datetime.utcfromtimestamp(
                dt64_to_unixtime(time.values)
            )
            formatted_time: str = raingrid_time.strftime("%Y-%m-%d %H:%M")
            file_name: str = raingrid_time.strftime("%Y-%m-%d_%H%M")

            if not os.path.exists(f"{self.outputs_raw_dir}/{file_name}.npy"):
                logger.debug(
                    "[WRITE] Saving raingrid %s for web output...", formatted_time
                )
                raingrid_links = calc_dataset.isel(time=t).cml_id.values.tolist()
                # get avg/max rain intensity value
                r_median_value = np.nanmedian(rain_grids[t])
                r_avg_value = np.nanmean(rain_grids[t])
                r_max_value = np.nanmax(rain_grids[t])

                if not self.skip_sql:
                    logger.debug(
                        "[WRITE] Writing raingrid's %s metadata into MariaDB...",
                        formatted_time,
                    )
                    self.sql_man.insert_raingrid(
                        time=raingrid_time,
                        links=raingrid_links,
                        file_name=f"{file_name}.png",
                        r_median=r_median_value,
                        r_avg=r_avg_value,
                        r_max=r_max_value,
                    )

                rain_grid = rain_grids[t]

                if self.is_crop_enabled:
                    logger.debug(
                        "[WRITE] Cropping raingrid %s to the GeoJSON polygon(s)...",
                        formatted_time,
                    )
                    rain_grid = mask_grid(rain_grid, x_grid, y_grid, prepared_polygons)

                logger.debug(
                    "[WRITE] Saving raingrid %s as PNG file...", formatted_time
                )
                ndarray_to_png(rain_grid, f"{self.output_dir}/{file_name}.png")

                logger.debug(
                    "[WRITE] Saving raingrid %s as raw numpy file...", formatted_time
                )
                save_ndarray_to_file(
                    rain_grid, f"{self.outputs_raw_dir}/{file_name}.npy"
                )

                logger.debug("[WRITE] Raingrid %s successfully saved.", formatted_time)

        logger.info("[WRITE] Saving raingrids locally -- DONE.")

    def _write_timeseries(
        self,
        calc_dataset: Dataset,
    ):
        """
        Write individual CML rain instensity timeseries into InfluxDB.
        :param calc_dataset: xarray Dataset with calculation data, containing CML timeseries with rain intensity
        :param np_last_time: last raingrid time in the database
        :param np_since_time: time since last realtime calculation start (overwritten by historic write)
        """

        points_to_write = []

        logger.info(
            "[WRITE: InfluxDB] Preparing rain timeseries from individual CMLs for writing into InfluxDB..."
        )

        # filtered = calc_dataset.where(calc_dataset.time > compare_time).dropna(
        #     dim="time", how="all"
        # )
        filtered = calc_dataset

        cmls_count = filtered.cml_id.size
        times_count = filtered.time.size

        if (cmls_count > 0) and (times_count > 0):
            for cml in range(cmls_count):
                for time in range(times_count):
                    points_to_write.append(
                        Point("telcorain")
                        .tag("cml_id", int(filtered.isel(cml_id=cml).cml_id))
                        .field(
                            "rain_intensity",
                            float(
                                filtered.isel(cml_id=cml)
                                .R.mean(dim="channel_id")
                                .isel(time=time)
                            ),
                        )
                        .time(
                            dt64_to_unixtime(filtered.isel(time=time).time.values),
                            write_precision=WritePrecision.S,
                        )
                    )
        if self.influx_wipe_thread is not None:
            logger.debug(
                "[WRITE: InfluxDB] Force write is active. Checking if InfluxDB wipe thread is done..."
            )
            self.influx_wipe_thread.join()
            logger.debug(
                "[WRITE: InfluxDB] Wipe thread is done. Proceeding with timeseries writing..."
            )
        logger.info(
            "[WRITE: InfluxDB] Writing rain timeseries from individual CMLs into database..."
        )
        self.influx_man.write_points(points_to_write, self.influx_man.BUCKET_OUT_CML)
        logger.info(
            "[WRITE: InfluxDB] Writing rain timeseries from individual CMLs -- DONE."
        )

    def push_results(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ):
        """
        Push the results of the real-time calculation into the database and outputs directory.
        :param rain_grids: list of 2D numpy arrays with rain intensity values
        :param x_grid: 2D numpy array of x coordinates
        :param y_grid: 2D numpy array of y coordinates
        :param calc_dataset: xarray Dataset with calculatt`ion data
        """
        # lock the influx manager to prevent start of new calculation before writing is finished
        # (lock is checked by the calculation starting mechanism, if the lock is active, calculation cannot start)
        self.influx_man.is_manager_locked = True
        if len(rain_grids) != len(calc_dataset.time):
            logger.error(
                "Cannot write raingrids into DB! Inconsistent count of rain grid frames "
                "(%d) and times in calculation dataset (%d)!",
                len(rain_grids),
                len(calc_dataset.time),
            )
            return

        if not os.path.exists("outputs_historic"):
            os.makedirs("outputs_historic")

        # if self.config["setting"]["compensate_historic"]:
        #     desired_start = self.cp["time"]["start"]
        #     # filter calc_dataset by time
        #     calc_dataset = calc_dataset.sel(time=slice(desired_start, None))
        #     # slice rain_grids to match the new dataset (assuming 1-to-1 time/sample mapping)
        #     time_len = calc_dataset.sizes["time"]
        #     rain_grids = rain_grids[-time_len:]

        # if not self.skip_sql:
        #     last_record = self.sql_man.get_last_raingrid()

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(self.outputs_raw_dir):
            os.mkdir(self.outputs_raw_dir)

        # I. RAINGRIDS SAVING LOCALLY AND TO MARIADB
        self._write_raingrids(
            rain_grids,
            x_grid,
            y_grid,
            calc_dataset,
        )
        del rain_grids

        # II. INDIVIDUAL CML TIMESERIES INTO INFLUXDB
        if not self.skip_influx:
            self._write_timeseries(calc_dataset)
        del calc_dataset

        # unlock the influx manager
        self.influx_man.is_manager_locked = False

    def start_push_results_thread(
        self,
        rain_grids: list[np.ndarray],
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        calc_dataset: Dataset,
    ) -> Thread:
        """
        Start a thread for pushing the results of the real-time calculation into the database and outputs directory.
        :param rain_grids: list of 2D numpy arrays with rain intensity values
        :param x_grid: 2D numpy array of x coordinates
        :param y_grid: 2D numpy array of y coordinates
        :param calc_dataset: xarray Dataset with calculation data
        :return: the thread object with the started activity
        """
        thread = Thread(
            target=self.push_results, args=(rain_grids, x_grid, y_grid, calc_dataset)
        )
        thread.start()
        return thread


def mask_grid(
    data_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    prepared_polygons: list[PreparedGeometry],
) -> np.ndarray:
    """
    Mask the 2D data grid with polygons. If a point is not within any of the polygons, it is set to NaN.

    :param data_grid: 2D ndarray data grid to be masked
    :param x_grid: 2D ndarray of x coordinates
    :param y_grid: 2D ndarray of y coordinates
    :param prepared_polygons: list of prepared polygons
    :return: masked 2D ndarray with NaN values outside the polygons
    """
    mask = np.vectorize(
        lambda lon, lat: any(
            polygon.contains(GeoPoint(lon, lat)) for polygon in prepared_polygons
        )
    )
    within_polygons = mask(x_grid, y_grid)
    data_grid[~within_polygons] = np.nan
    return data_grid


def _get_color(value: float) -> tuple[int, int, int, int]:
    """
    Get RGBA color tuple based on the rain intensity value.
    The color scale is identical to the CHMI rain scale colorbar. The rain intensity values are in mm/h, the scale is
    derived from the CHMI radar scale, where dBZ have been converted to mm/h using the Marshall-Palmer formula:
    https://rdrr.io/github/potterzot/kgRainPredictR/man/marshall_palmer.html

    [dBZ]  [mm/h]    [RGBA]
    4      0.064842  (57, 0, 112, 255)
    8      0.115307  (47, 1, 169, 255)
    12     0.205048  (0, 0, 252, 255)
    16     0.364633  (0, 108, 192, 255)
    20     0.648420  (0, 160, 0, 255)
    24     1.153072  (0, 188, 0, 255)
    28     2.050483  (52, 216, 0, 255)
    32     3.646332  (156, 220, 0, 255)
    36     6.484198  (224, 220, 0, 255)
    40     11.53072  (252, 176, 0, 255)
    44     20.50483  (252, 132, 0, 255)
    48     36.46332  (252, 88, 0, 255)
    52     64.84198  (252, 0, 0, 255)
    56     115.3072  (160, 0, 0, 255)

    :param value: rain intensity value
    :return: RGBA color tuple, NaN and values below 0.1 are transparent
    """
    if np.isnan(value) or value < 0.1:
        return 0, 0, 0, 0  # transparent
    elif 0.1 <= value < 0.115307:
        return 57, 0, 112, 255
    elif 0.115307 <= value < 0.205048:
        return 47, 1, 169, 255
    elif 0.205048 <= value < 0.364633:
        return 0, 0, 252, 255
    elif 0.364633 <= value < 0.648420:
        return 0, 108, 192, 255
    elif 0.648420 <= value < 1.153072:
        return 0, 160, 0, 255
    elif 1.153072 <= value < 2.050483:
        return 0, 188, 0, 255
    elif 2.050483 <= value < 3.646332:
        return 52, 216, 0, 255
    elif 3.646332 <= value < 6.484198:
        return 156, 220, 0, 255
    elif 6.484198 <= value < 11.53072:
        return 224, 220, 0, 255
    elif 11.53072 <= value < 20.50483:
        return 252, 176, 0, 255
    elif 20.50483 <= value < 36.46332:
        return 252, 132, 0, 255
    elif 36.46332 <= value < 64.84198:
        return 252, 88, 0, 255
    elif 64.84198 <= value < 115.3072:
        return 252, 0, 0, 255
    elif value >= 115.3072:
        return 160, 0, 0, 255
    else:
        return 0, 0, 0, 0  # default: transparent


def ndarray_to_png(array: np.ndarray, output_path: str):
    """
    Convert 2D numpy array into a PNG image.
    :param array: 2D numpy array with rain intensity values
    :param output_path: path to the output PNG image
    """
    height, width = array.shape
    image = Image.new("RGBA", (width, height))
    pixels = image.load()

    # assign colors to pixels based on the values in the ndarray
    for i in range(width):
        for j in range(height):
            # save pixel at height-j-1 Y position due to vertical flip
            pixels[i, height - j - 1] = _get_color(array[j, i])

    try:
        image.save(output_path, "PNG")
    except Exception as error:
        logger.error("Cannot save PNG image: %s", error)


def save_ndarray_to_file(array: np.ndarray, output_path: str):
    """
    Save a 2D numpy ndarray to a local file.
    :param array: 2D numpy array to be saved
    :param output_path: Path to the file where the array will be saved
    """
    try:
        np.save(output_path, array)
    except Exception as error:
        logger.error('Cannot save ndarray to file "%s": %s', output_path, error)


def read_from_ndarray_file(input_path: str) -> np.ndarray:
    """
    Read a 2D numpy ndarray from a local file.

    Parameters:
    :param input_path: Path to the saved ndarray file

    Returns:
    array: The 2D numpy ndarray read from the file.
    """
    try:
        array: np.ndarray = np.round(np.load(input_path), decimals=3)
    except FileNotFoundError as error:
        logger.error(
            'Cannot read stored ndarray file "%s": File not found.', input_path
        )
        raise error
    except Exception as error:
        logger.error('Cannot read stored ndarray file "%s": %s', input_path, error)
        raise error

    return array


def read_value_from_ndarray_file(
    input_path: str,
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    total_rows: int,
    total_cols: int,
) -> np.number:
    """
    Read a value from a saved 2D numpy ndarray local file based on given geographic coordinates.

    Parameters:
    :param input_path: Path to the saved ndarray file
    :param x: Longitude value to read
    :param y: Latitude value to read
    :param x_min: Minimum longitude (array border)
    :param x_max: Maximum longitude (array border)
    :param y_min: Minimum latitude (array border)
    :param y_max: Maximum latitude (array border)
    :param total_rows: Total number of rows in the array (vertical resolution)
    :param total_cols: Total number of columns in the array (horizontal resolution)

    Returns:
    value: The value at the specified geographic coordinates in the array.
    """
    try:
        array: np.ndarray = np.load(input_path)
    except FileNotFoundError as error:
        logger.error(
            'Cannot read stored ndarray file "%s": File not found.', input_path
        )
        raise error
    except Exception as error:
        logger.error('Cannot read stored ndarray file "%s": %s', input_path, error)
        raise error

    x_step = (x_max - x_min) / (total_cols - 1)
    y_step = (y_max - y_min) / (total_rows - 1)

    # calculate the closest row and column indices
    col = round((x - x_min) / x_step)
    row = round((y - y_min) / y_step)

    # ensure indices are within array bounds
    col = min(max(col, 0), total_cols - 1)
    row = min(max(row, 0), total_rows - 1)

    return array[row, col]


def purge_raw_outputs(config: dict):
    """
    Purge the .npy files in the raw outputs directory.
    """
    try:
        raw_outputs_dir = config["directories"]["outputs_raw"]
        for file in os.listdir(raw_outputs_dir):
            file_path = os.path.join(raw_outputs_dir, file)
            try:
                if os.path.isfile(file_path):
                    if file_path.endswith(".npy"):
                        os.unlink(file_path)
                        logger.debug('[DEVMODE] NPY file "%s" deleted.', file_path)
                    else:
                        logger.warning(
                            '[DEVMODE] Cannot delete file "%s": It is not a NPY file.',
                            file_path,
                        )
                elif os.path.isdir(file_path):
                    logger.warning(
                        '[DEVMODE] Raw outputs directory contains another directory "%s", which cannot be deleted.',
                        file_path,
                    )
            except Exception as error:
                logger.warning('Cannot delete file "%s": %s', file_path, error)
        logger.info("[DEVMODE] Raw outputs directory erased.")
    except Exception as error:
        logger.error("Cannot purge raw outputs directory: %s", error)


def purge_web_outputs(config: dict):
    """
    Purge the .png files in the web outputs directory.
    """
    try:
        web_outputs_dir = config["directories"]["outputs_raw"]
        for file in os.listdir(web_outputs_dir):
            file_path = os.path.join(web_outputs_dir, file)
            try:
                if os.path.isfile(file_path):
                    if file_path.endswith(".png"):
                        os.unlink(file_path)
                        logger.debug('[DEVMODE] PNG file "%s" deleted.', file_path)
                    else:
                        logger.warning(
                            '[DEVMODE] Cannot delete file "%s": It is not a PNG file.',
                            file_path,
                        )
                elif os.path.isdir(file_path):
                    logger.warning(
                        '[DEVMODE] Raw outputs directory contains another directory "%s", which cannot be deleted.',
                        file_path,
                    )
            except Exception as error:
                logger.warning('Cannot delete file "%s": %s', file_path, error)
        logger.info("[DEVMODE] Web outputs directory erased.")
    except Exception as error:
        logger.error("Cannot purge web outputs directory: %s", error)
