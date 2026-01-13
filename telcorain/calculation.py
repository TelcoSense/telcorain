from datetime import datetime
from typing import Optional

import numpy as np
import xarray as xr

from telcorain.database.influx_manager import InfluxManager
from telcorain.handlers import logger
from telcorain.dataprocessing import (
    convert_to_link_datasets,
    load_data_from_influxdb,
)
from telcorain.procedures.exceptions import (
    ProcessingException,
    RaincalcException,
    RainfieldsGenException,
)
from telcorain.procedures.rain import rain_calculation
from telcorain.procedures.rain.rainfields_generation import generate_rainfields
from telcorain.helpers import measure_time, MwLink


class Calculation:
    """
    Unified Calculation (realtime / historic).

    x_grid, y_grid:
        - If interp.use_mercator = False: lon/lat in degrees.
        - If interp.use_mercator = True:  EPSG:3857 coordinates in metres.
    """

    def __init__(
        self,
        influx_man: InfluxManager,
        links: dict[int, MwLink],
        selection: dict[int, int],
        config: dict,
        *,
        is_historic: bool = False,
        results_id: Optional[int] = None,
    ):
        self.influx_man = influx_man
        self.links = links
        self.selection = selection
        self.config = config

        # realtime-specific counters
        self.realtime_runs = 0
        self.thousands_runs = 0

        # historic mode settings
        self.is_historic = is_historic
        self.results_id = results_id

        # persistent grids/state
        self.rain_grids: list[np.ndarray] = []
        self.rain_grids_sum: list[np.ndarray] = []
        self.x_grid: Optional[np.ndarray] = None
        self.y_grid: Optional[np.ndarray] = None
        self.calc_data_steps: Optional[xr.Dataset] = None
        self.last_time: np.datetime64 = np.datetime64(datetime.min)

    # =====================================================================
    # RUN
    # =====================================================================

    @measure_time
    def run(self, realtime_timewindow: str = "1d"):
        """
        Unified RUN function that behaves either like realtime or historic calculation.
        """

        if self.is_historic:
            log_run_id = "Historic run"
            further_info = "Historic"
        else:
            self.realtime_runs += 1
            log_run_id = f"RUN: {self.realtime_runs}"
            further_info = "Realtime"

        logger.info(
            f"[{log_run_id}] {further_info} rainfall calculation procedure started."
        )

        # =====================================================================
        # 1. LOAD DATA FROM INFLUX
        # =====================================================================
        try:
            df, missing_links, ips = load_data_from_influxdb(
                influx_man=self.influx_man,
                config=self.config,
                selected_links=self.selection,
                links=self.links,
                log_run_id=log_run_id,
                realtime=not self.is_historic,
                realtime_timewindow=(
                    realtime_timewindow if not self.is_historic else "1d"
                ),
            )

            calc_data: list[xr.Dataset] = convert_to_link_datasets(
                selected_links=self.selection,
                links=self.links,
                df=df,
                missing_links=missing_links,
                log_run_id=log_run_id,
            )
            del df

        except ProcessingException:
            return

        # =====================================================================
        # 2. COMPUTE RAIN RATES
        # =====================================================================
        try:
            calc_data = rain_calculation.get_rain_rates(
                calc_data=calc_data,
                config=self.config,
                ips=ips,
                log_run_id=log_run_id,
            )
        except RaincalcException:
            return

        # =====================================================================
        # 3. RAINFIELDS (Unified)
        # =====================================================================
        try:
            result = generate_rainfields(
                calc_data=calc_data,
                config=self.config,
                rain_grids=self.rain_grids,
                is_historic=self.is_historic,
                realtime_runs=self.realtime_runs,
                last_time=self.last_time,
                log_run_id=log_run_id,
            )

            if not self.is_historic:
                (
                    self.rain_grids,
                    self.rain_grids_sum,
                    self.calc_data_steps,
                    self.x_grid,
                    self.y_grid,
                    self.realtime_runs,
                    self.last_time,
                ) = result
            else:
                (
                    self.rain_grids,
                    self.rain_grids_sum,
                    self.calc_data_steps,
                    self.x_grid,
                    self.y_grid,
                ) = result

        except RainfieldsGenException:
            return

        # =====================================================================
        # 4. REALTIME housekeeping
        # =====================================================================
        if not self.is_historic:
            # refresh full data after 1000 runs
            self.force_data_refresh = self.realtime_runs % 1000 == 0

            # prevent overflow
            if self.realtime_runs == 99999:
                self.realtime_runs = 1
                self.thousands_runs += 1
                logger.info(
                    "Refreshing realtime_runs to 1 after 99999 runs. "
                    f"No. of thousand runs: {self.thousands_runs}"
                )
