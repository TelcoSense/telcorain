from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
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
            # Keep time bounds for optional high-resolution queries (e.g. CNN wet/dry on 30 s)
            df_time_min = None
            df_time_max = None
            try:
                if df is not None and not df.empty and "_time" in df.columns:
                    df_time_min = pd.to_datetime(df["_time"].min(), utc=True)
                    df_time_max = pd.to_datetime(df["_time"].max(), utc=True)
            except Exception:
                df_time_min = None
                df_time_max = None
            del df

        except ProcessingException:
            return

        # =====================================================================
        # 1b. Optional: compute *custom* CNN wet/dry on 30 s data, then map to 10 min
        # =====================================================================
        try:
            wd_cfg = self.config.get("wet_dry", {})
            if bool(wd_cfg.get("is_mlp_enabled", False)) and wd_cfg.get("cnn_model") not in [None, "", "polz"]:
                from telcorain.procedures.wet_dry.cnn_resample import (
                    compute_wet_mask_10min_from_30s,
                )
                import pandas as pd

                # If bounds not available, fall back to config range.
                start_utc = (df_time_min.to_pydatetime() if df_time_min is not None else self.config["time"]["start"])
                end_utc = (df_time_max.to_pydatetime() if df_time_max is not None else self.config["time"]["end"])

                # Query 30 s data just for wet/dry classification.
                df30 = self.influx_man.query_units_seconds(
                    ips=ips,
                    start=start_utc,
                    end=end_utc,
                    interval_seconds=30,
                    rolling_values=None,
                    compensate_historic=False,
                )

                if df30 is not None and not df30.empty:
                    calc_data_30s = convert_to_link_datasets(
                        selected_links=self.selection,
                        links=self.links,
                        df=df30,
                        missing_links=[],
                        log_run_id=f"{log_run_id} CNN30s",
                    )

                    # Map by cml_id
                    by_id_30s = {int(ds.cml_id.values): ds for ds in calc_data_30s}
                    for i, ds10 in enumerate(calc_data):
                        cml_id = int(ds10.cml_id.values)
                        ds30 = by_id_30s.get(cml_id)
                        if ds30 is None:
                            continue

                        wet10 = compute_wet_mask_10min_from_30s(
                            ds30,
                            ds10.time,
                            model_param_dir=wd_cfg["cnn_model_name"],
                            sample_size=60,
                            threshold=0.5,
                            target_rule="max",
                            fillna_dry=True,
                        )

                        # Attach, so rain_calculation can reuse it.
                        ds10["wet"] = (("time",), wet10.astype(bool))
                        calc_data[i] = ds10
        except Exception as e:
            logger.warning(
                f"[{log_run_id}] Custom CNN 30 s wet/dry precomputation failed, falling back to in-pipeline logic. Error: {e}"
            )

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
                rain_grids_sum=self.rain_grids_sum,
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
