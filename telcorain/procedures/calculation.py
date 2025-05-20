from datetime import datetime, timedelta, timezone
from typing import Union
from os.path import exists

import pickle
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
from telcorain.procedures.utils.helpers import measure_time


class Calculation:
    def __init__(
        self,
        influx_man: InfluxManager,
        links: dict[int, MwLink],
        selection: dict[int, int],
        cp: dict,
        config: dict,
    ):

        self.influx_man: InfluxManager = influx_man
        self.links: dict[int, MwLink] = links
        self.selection: dict[int, int] = selection
        self.realtime_runs: int = 0
        self.thousands_runs: int = 0

        # calculation parameters dictionary
        self.cp = cp
        # config parameters dictionary
        self.config = config

        # store raingrids for possible next iteration (no need for repeated generating in realtime?)
        self.rain_grids: list[np.ndarray] = []
        self.x_grid: np.ndarray = None
        self.y_grid: np.ndarray = None
        self.calc_data_steps = None
        self.last_time: np.datetime64 = np.datetime64(datetime.min)

    @measure_time
    def run(self, realtime_timewindow: str = "1d"):
        self.realtime_runs += 1
        log_run_id = "RUN: " + str(self.realtime_runs)
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
                realtime=self.cp["realtime"]["is_realtime"],
                realtime_timewindow=realtime_timewindow,
            )

            # Merge influx data with metadata into datasets, resolve Tx power assignment to correct channel
            calc_data: list[xr.Dataset] = data_preprocessing.convert_to_link_datasets(
                selected_links=self.selection,
                links=self.links,
                influx_data=influx_data,
                missing_links=missing_links,
                log_run_id=log_run_id,
            )

            # je to safe? neni lepsi drzet influx data v ramce? asi zalezi na nastaveni...
            if self.cp["realtime"]["realtime_optimization"]:
                with open("temp_data/temp_data.pkl", "wb") as f:
                    pickle.dump(influx_data, f)
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
            (
                self.rain_grids,
                self.calc_data_steps,
                self.x_grid,
                self.y_grid,
                self.realtime_runs,
                self.last_time,
            ) = rainfields_generation.generate_rainfields(
                calc_data=calc_data,
                cp=self.cp,
                rain_grids=self.rain_grids,
                realtime_runs=self.realtime_runs,
                last_time=self.last_time,
                log_run_id=log_run_id,
            )
        except RainfieldsGenException:
            return

        logger.info("[%s] Rainfall calculation procedure ended.", log_run_id)

        # once per 1000 runs, update the whole influx_data entry so it does load all currently available CMLs again
        if self.realtime_runs % 1000 == 0:
            self.force_data_refresh = True
        else:
            self.force_data_refresh = False

        # we do not want realtime_runs to overflow
        if self.realtime_runs == 99999:
            self.realtime_runs = 1
            self.thousands_runs += 1
            logger.info(
                f"Refreshing realtime_runs to 1 after 99999 runs. No. of thousand runs: {self.thousands_runs}"
            )


class CalculationHistoric:
    def __init__(
        self,
        influx_man: InfluxManager,
        results_id: int,
        links: dict[int, MwLink],
        selection: dict[int, int],
        cp: dict,
        compensate_historic: bool,
    ):

        self.influx_man: InfluxManager = influx_man
        self.results_id: int = results_id
        self.links: dict[int, MwLink] = links
        self.selection: dict[int, int] = selection
        self.compensate_historic = compensate_historic

        # calculation parameters dictionary
        self.cp = cp
        # store raingrids for possible next iteration (no need for repeated generating in realtime)
        self.rain_grids: list[np.ndarray] = []
        self.x_grid: np.ndarray = None
        self.y_grid: np.ndarray = None
        self.calc_data_steps = None

    @measure_time
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
                realtime=False,
                compensate_historic=self.compensate_historic,
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
                    log_run_id=log_run_id,
                )
            )
        except RainfieldsGenException:
            return

        logger.info("[%s] Rainfall calculation procedure ended.", log_run_id)
