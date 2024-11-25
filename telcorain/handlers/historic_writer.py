from datetime import datetime

import numpy as np
import xarray as xr

from ..database.influx_manager import influx_man
from ..database.sql_manager import sql_man
from ..procedures.data.data_loading import load_data_from_influxdb
from ..procedures.data.data_preprocessing import convert_to_link_datasets
from ..procedures.rain.rain_calculation import get_rain_rates
from ..procedures.rain.rainfields_generation import generate_rainfields


class HistoricWriter:
    def __init__(self, calculation_params: dict):

        self.calculation_params = calculation_params
        self.sql_man = sql_man
        self.influx_man = influx_man
        self.links = sql_man.load_metadata()
        self.selected_links = {237: 3, 238: 3}
        self.last_time: np.datetime64 = np.datetime64(datetime.min)
        self.rain_grids: list[np.ndarray] = []
        self.realtime_runs = 0

    def calculate_rain_grids(self):
        influx_data, missing_links, ips = load_data_from_influxdb(
            influx_man=self.influx_man,
            cp=self.calculation_params,
            selected_links=self.selected_links,
            links=self.links,
        )

        calc_data: list[xr.Dataset] = convert_to_link_datasets(
            selected_links=self.selected_links,
            links=self.links,
            influx_data=influx_data,
            missing_links=missing_links,
        )

        calc_data: list[xr.Dataset] = get_rain_rates(
            calc_data=calc_data,
            cp=self.calculation_params,
            ips=ips,
        )

        self.rain_grids, self.realtime_runs, self.last_time = generate_rainfields(
            calc_data=calc_data,
            cp=self.calculation_params,
            rain_grids=self.rain_grids,
            realtime_runs=self.realtime_runs,
            last_time=self.last_time,
        )
