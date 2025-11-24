import traceback
from typing import Any, Optional

import numpy as np
import xarray as xr
from pycomlink.spatial.interpolator import IdwKdtreeInterpolator

from telcorain.handlers import logger
from telcorain.procedures.exceptions import RainfieldsGenException
from telcorain.helpers import measure_time


@measure_time
def generate_rainfields(
    calc_data: list[xr.Dataset],
    config: dict[str, Any],
    rain_grids: list[np.ndarray],
    *,
    is_historic: bool = False,
    realtime_runs: int = 1,
    last_time: Optional[np.datetime64] = None,
    log_run_id: str = "default",
):
    """
    Historic mode:
      - ignore realtime_runs, last_time
      - does NOT delete old frames
      - does NOT return realtime_runs or last_time

    Realtime mode:
      - keeps original deletion of first N frames where needed
      - returns realtime_runs and last_time

    Return values:
       Realtime:
          (rain_grids, calc_data_steps, x_grid, y_grid, realtime_runs, last_time)
       Historic:
          (rain_grids, calc_data_steps, x_grid, y_grid)
    """

    try:
        logger.info("[%s] Generating rainfields...", log_run_id)

        if not calc_data:
            logger.warning("[%s] Empty calc_data, nothing to interpolate.", log_run_id)
            if is_historic:
                return rain_grids, None, None, None
            else:
                return rain_grids, None, None, None, realtime_runs, last_time

        # ------------------------------------------------------------------
        # 0) Concatenate all links once and precompute geometry
        # ------------------------------------------------------------------
        # ds_all has dims: cml_id, channel_id, time
        ds_all = xr.concat(calc_data, dim="cml_id")

        # Compute centers once (broadcast over time automatically)
        ds_all = ds_all.assign(
            lat_center=(ds_all.site_a_latitude + ds_all.site_b_latitude) / 2,
            lon_center=(ds_all.site_a_longitude + ds_all.site_b_longitude) / 2,
        )

        # ------------------------------------------------------------------
        # 1) Create IDW interpolator & target grid (only once)
        # ------------------------------------------------------------------
        interpolator = IdwKdtreeInterpolator(
            nnear=config["interp"]["idw_near"],
            p=config["interp"]["idw_power"],
            exclude_nan=True,
            max_distance=config["interp"]["idw_dist"],
        )

        x_coords = np.arange(
            config["limits"]["x_min"],
            config["limits"]["x_max"],
            config["interp"]["interp_res"],
        )
        y_coords = np.arange(
            config["limits"]["y_min"],
            config["limits"]["y_max"],
            config["interp"]["interp_res"],
        )
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # N.B. x, y do not depend on time; re-use as NumPy arrays in loops
        x_sites = ds_all.lon_center.values
        y_sites = ds_all.lat_center.values

        # ------------------------------------------------------------------
        # 2) Overall accumulated rainfall field (1h means, sum over time)
        # ------------------------------------------------------------------
        logger.debug(
            "[%s] Resampling rain values for rainfall overall map...", log_run_id
        )

        # 1h resample from concatenated dataset
        calc_data_1h = ds_all.R.resample(time="1H", label="right").mean().to_dataset()

        logger.debug(
            "[%s] Interpolating spatial data for rainfall overall map...", log_run_id
        )

        # Mean over channels, then sum in time => total accumulation
        z_overall = (
            calc_data_1h.R.mean(dim="channel_id").sum(dim="time").values
        )  # shape: (cml_id,)

        rain_grid = interpolator(
            x=x_sites,
            y=y_sites,
            z=z_overall,
            xgrid=x_grid,
            ygrid=y_grid,
        )
        # (rain_grid is computed for consistency with original code; if needed,
        # you can store or return it from here.)

        # ------------------------------------------------------------------
        # 3) Time-step rainfall fields for animation
        # ------------------------------------------------------------------
        if not config["raingrids"]["is_only_overall"]:

            logger.debug(
                "[%s] Resampling data for rainfall animation maps...", log_run_id
            )

            ts = config["time"]["output_step"]  # [minutes]
            base_step = config["time"]["step"]  # [minutes]

            # Decide source dataset for animation time steps
            if ts == 60:
                # use 1h resample
                calc_data_steps = calc_data_1h
            elif ts > base_step:
                # resample from base ds_all to desired step
                calc_data_steps = (
                    ds_all.R.resample(time=f"{ts}T", label="right").mean().to_dataset()
                )
            elif ts == base_step:
                # use base-resolution data as is
                calc_data_steps = ds_all
            else:
                raise ValueError("Invalid value of output_step")

            # convert mm/h → mm if requested
            if config["raingrids"]["is_output_total"]:
                time_ratio = 60.0 / float(ts)  # hours per step
                calc_data_steps["R"] = calc_data_steps.R / time_ratio

            logger.debug(
                "[%s] Interpolating spatial data for rainfall animation maps...",
                log_run_id,
            )

            # lat_center / lon_center already present on ds_all; xarray will
            # carry them into resampled calc_data_steps (no need to recompute)

            # Precompute z(t) as NumPy for fast looping
            # shape: (cml_id, time)
            z_all = calc_data_steps.R.mean(dim="channel_id").values
            times = calc_data_steps.time.values
            min_rain = config["raingrids"]["min_rain_value"]

            grids_to_del = 0

            for i in range(z_all.shape[1]):
                z_t = z_all[:, i]

                grid = interpolator(
                    x=x_sites,
                    y=y_sites,
                    z=z_t,
                    xgrid=x_grid,
                    ygrid=y_grid,
                )
                grid[grid < min_rain] = 0.0
                rain_grids.append(grid)

                if not is_historic:
                    last_time = times[i]
                    if realtime_runs > 1:
                        grids_to_del += 1

            # Maintain sliding window (realtime only)
            if not is_historic and grids_to_del > 0:
                # delete oldest frames in one slice
                del rain_grids[:grids_to_del]

            # ------------------------------------------------------------------
            # 4) Return in the original shapes
            # ------------------------------------------------------------------
            if is_historic:
                return rain_grids, calc_data_steps, x_grid, y_grid
            else:
                return (
                    rain_grids,
                    calc_data_steps,
                    x_grid,
                    y_grid,
                    realtime_runs,
                    last_time,
                )

        # If only overall field is requested, still return something consistent
        # with the previous API (no per-step fields).
        if is_historic:
            return rain_grids, calc_data_1h, x_grid, y_grid
        else:
            return rain_grids, calc_data_1h, x_grid, y_grid, realtime_runs, last_time

    except BaseException as error:
        logger.error(
            "[%s] Error during rainfields generation: %s %s.\nCalculation aborted.",
            log_run_id,
            type(error),
            error,
        )
        traceback.print_exc()
        raise RainfieldsGenException("Error during rainfall fields generation")
