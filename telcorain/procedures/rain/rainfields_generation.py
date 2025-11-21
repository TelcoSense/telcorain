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

        # **********************************************************************
        # ***** FIRST PART: Calculate overall rainfall accumulation field ******
        # **********************************************************************

        logger.debug(
            "[%s] Resampling rain values for rainfall overall map...", log_run_id
        )

        # resample values to 1h means
        calc_data_1h = xr.concat(
            objs=[cml.R.resample(time="1H", label="right").mean() for cml in calc_data],
            dim="cml_id",
        ).to_dataset()

        logger.debug(
            "[%s] Interpolating spatial data for rainfall overall map...", log_run_id
        )

        # Compute midpoints for interpolation
        calc_data_1h["lat_center"] = (
            calc_data_1h.site_a_latitude + calc_data_1h.site_b_latitude
        ) / 2
        calc_data_1h["lon_center"] = (
            calc_data_1h.site_a_longitude + calc_data_1h.site_b_longitude
        ) / 2

        # Create IDW interpolator
        interpolator = IdwKdtreeInterpolator(
            nnear=config["interp"]["idw_near"],
            p=config["interp"]["idw_power"],
            exclude_nan=True,
            max_distance=config["interp"]["idw_dist"],
        )

        # Generate coordinate grid
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

        # Compute overall accumulated rainfall
        rain_grid = interpolator(
            x=calc_data_1h.lon_center,
            y=calc_data_1h.lat_center,
            z=calc_data_1h.R.mean(dim="channel_id").sum(dim="time"),
            xgrid=x_grid,
            ygrid=y_grid,
        )

        # compute start and end timestamps (preserved but unused)
        data_start = calc_data[0].time.min()
        data_end = calc_data[0].time.max()
        for link in calc_data:
            times = link.time.values
            data_start = min(data_start, times.min())
            data_end = max(data_end, times.max())

        # *******************************************************************
        # ***** SECOND PART: Calculate individual fields for animation ******
        # *******************************************************************

        if not config["raingrids"]["is_only_overall"]:

            logger.debug(
                "[%s] Resampling data for rainfall animation maps...", log_run_id
            )

            # resample data to desired resolution
            ts = config["time"]["output_step"]
            base_step = config["time"]["step"]

            if ts == 60:
                calc_data_steps = calc_data_1h
            elif ts > base_step:
                calc_data_steps = xr.concat(
                    objs=[
                        cml.R.resample(time=f"{ts}T", label="right").mean()
                        for cml in calc_data
                    ],
                    dim="cml_id",
                ).to_dataset()
            elif ts == base_step:
                calc_data_steps = xr.concat(calc_data, dim="cml_id")
            else:
                raise ValueError("Invalid value of output_steps")

            del calc_data

            # convert mm/h to mm if needed
            if config["raingrids"]["is_output_total"]:
                time_ratio = 60 / ts
                calc_data_steps["R"] = calc_data_steps.R / time_ratio

            logger.debug(
                "[%s] Interpolating spatial data for rainfall animation maps...",
                log_run_id,
            )

            # compute centers if not already done
            if ts != 60:
                calc_data_steps["lat_center"] = (
                    calc_data_steps.site_a_latitude + calc_data_steps.site_b_latitude
                ) / 2
                calc_data_steps["lon_center"] = (
                    calc_data_steps.site_a_longitude + calc_data_steps.site_b_longitude
                ) / 2

            # Interpolation loop
            grids_to_del = 0
            for i in range(calc_data_steps.time.size):
                grid = interpolator(
                    x=calc_data_steps.lon_center,
                    y=calc_data_steps.lat_center,
                    z=calc_data_steps.R.mean(dim="channel_id").isel(time=i),
                    xgrid=x_grid,
                    ygrid=y_grid,
                )
                grid[grid < config["raingrids"]["min_rain_value"]] = 0
                rain_grids.append(grid)

                if is_historic:
                    pass
                else:
                    last_time = calc_data_steps.time[i].values
                    if realtime_runs > 1:
                        grids_to_del += 1

            # delete oldest frames (realtime only)
            if not is_historic:
                for _ in range(grids_to_del):
                    del rain_grids[0]

            # cleanup
            if calc_data_1h is not None:
                del calc_data_1h

            # RETURN VALUES
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

    except BaseException as error:
        logger.error(
            "[%s] Error during rainfields generation: %s %s.\nCalculation aborted.",
            log_run_id,
            type(error),
            error,
        )
        traceback.print_exc()
        raise RainfieldsGenException("Error during rainfall fields generation")
