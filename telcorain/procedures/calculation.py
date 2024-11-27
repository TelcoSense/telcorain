from datetime import datetime
from typing import Union

import numpy as np
import xarray as xr

from telcorain.database.influx_manager import InfluxManager
from telcorain.database.models.mwlink import MwLink
from telcorain.handlers.logging_handler import logger
from telcorain.procedures.data import data_loading, data_preprocessing
from telcorain.procedures.exceptions import (
    ProcessingException,
    RaincalcException,
    RainfieldsGenException,
)
from telcorain.procedures.rain import rain_calculation, rainfields_generation


class Calculation:
    def __init__(
        self,
        influx_man: InfluxManager,
        results_id: int,
        links: dict[int, MwLink],
        selection: dict[int, int],
        cp: dict,
    ):

        self.influx_man: InfluxManager = influx_man
        self.results_id: int = results_id
        self.links: dict[int, MwLink] = links
        self.selection: dict[int, int] = selection

        # calculation parameters dictionary
        self.cp = cp

        # run counter in case of realtime calculation
        self.realtime_runs: int = 0

        # store raingrids for possible next iteration (no need for repeated generating in realtime)
        self.rain_grids: list[np.ndarray] = []
        self.last_time: np.datetime64 = np.datetime64(datetime.min)

    def run(self):
        self.realtime_runs += 1
        if self.cp["realtime"]["is_realtime"]:
            log_run_id = (
                "CALC ID: " + str(self.results_id) + ", RUN: " + str(self.realtime_runs)
            )
        else:
            log_run_id = "CALC ID: " + str(self.results_id)

        logger.info("[%s] Rainfall calculation procedure started.", log_run_id)

        try:
            # Gather data from InfluxDB
            influx_data: dict[str, Union[dict[str, dict[datetime, float]], str]]
            missing_links: list[int]
            ips: list[str]
            influx_data, missing_links, ips = data_loading.load_data_from_influxdb(
                influx_man=self.influx_man,
                cp=self.cp,
                selected_links=self.selection,
                links=self.links,
                log_run_id=log_run_id,
            )

            # Merge influx data with metadata into datasets, resolve Tx power assignment to correct channel
            calc_data: list[xr.Dataset] = data_preprocessing.convert_to_link_datasets(
                selected_links=self.selection,
                links=self.links,
                influx_data=influx_data,
                missing_links=missing_links,
                log_run_id=log_run_id,
            )
            del influx_data
        except ProcessingException:
            return

        try:
            # Obtain rain rates and store them in the calc_data
            calc_data: list[xr.Dataset] = rain_calculation.get_rain_rates(
                calc_data=calc_data,
                cp=self.cp,
                ips=ips,
                log_run_id=log_run_id,
            )
        except RaincalcException:
            return

        try:
            # Generate rainfields (resample rain rates and interpolate them to a grid)
            self.rain_grids, self.realtime_runs, self.last_time = (
                rainfields_generation.generate_rainfields(
                    calc_data=calc_data,
                    cp=self.cp,
                    rain_grids=self.rain_grids,
                    realtime_runs=self.realtime_runs,
                    last_time=self.last_time,
                    log_run_id=log_run_id,
                )
            )
        except RainfieldsGenException:
            return

        logger.info("[%s] Rainfall calculation procedure ended.", log_run_id)


class CalculationHistoric:
    def __init__(
        self,
        influx_man: InfluxManager,
        results_id: int,
        links: dict[int, MwLink],
        selection: dict[int, int],
        cp: dict,
    ):

        self.influx_man: InfluxManager = influx_man
        self.results_id: int = results_id
        self.links: dict[int, MwLink] = links
        self.selection: dict[int, int] = selection

        # calculation parameters dictionary
        self.cp = cp
        # store raingrids for possible next iteration (no need for repeated generating in realtime)
        self.rain_grids: list[np.ndarray] = []
        self.x_grid: np.ndarray = None
        self.y_grid: np.ndarray = None
        self.calc_data_steps = None

    def run(self):
        log_run_id = "CALC ID: " + str(self.results_id)
        logger.info("[%s] Rainfall calculation procedure started.", log_run_id)

        try:
            # Gather data from InfluxDB
            influx_data: dict[str, Union[dict[str, dict[datetime, float]], str]]
            missing_links: list[int]
            ips: list[str]
            influx_data, missing_links, ips = data_loading.load_data_from_influxdb(
                influx_man=self.influx_man,
                cp=self.cp,
                selected_links=self.selection,
                links=self.links,
                log_run_id=log_run_id,
            )

            # Merge influx data with metadata into datasets, resolve Tx power assignment to correct channel
            calc_data: list[xr.Dataset] = data_preprocessing.convert_to_link_datasets(
                selected_links=self.selection,
                links=self.links,
                influx_data=influx_data,
                missing_links=missing_links,
                log_run_id=log_run_id,
            )
            del influx_data
        except ProcessingException:
            return

        try:
            # Obtain rain rates and store them in the calc_data
            calc_data: list[xr.Dataset] = rain_calculation.get_rain_rates(
                calc_data=calc_data,
                cp=self.cp,
                ips=ips,
                log_run_id=log_run_id,
            )
        except RaincalcException:
            return

        try:
            # Generate rainfields (resample rain rates and interpolate them to a grid)
            self.rain_grids, self.calc_data_steps, self.x_grid, self.y_grid = (
                rainfields_generation.generate_rainfields_historic(
                    calc_data=calc_data,
                    cp=self.cp,
                    rain_grids=self.rain_grids,
                )
            )
        except RainfieldsGenException:
            return

        logger.info("[%s] Rainfall calculation procedure ended.", log_run_id)
