import traceback
from typing import Any

import numpy as np
import xarray as xr
from pycomlink.spatial.interpolator import IdwKdtreeInterpolator

from telcorain.handlers.logging_handler import logger
from telcorain.procedures.exceptions import RainfieldsGenException
from telcorain.procedures.utils.helpers import measure_time


# @measure_time
# def generate_rainfields(
#     calc_data: list[xr.Dataset],
#     cp: dict[str, Any],
#     rain_grids: list[np.ndarray],
#     realtime_runs: int,
#     last_time: np.datetime64,
#     log_run_id: str = "default",
# ) -> tuple[list[np.ndarray], int, np.datetime64]:
#     try:
#         logger.info("[%s] Generating rainfields...", log_run_id)
#         # **********************************************************************
#         # ***** FIRST PART: Calculate overall rainfall accumulation field ******
#         # **********************************************************************

#         logger.debug(
#             "[%s] Resampling rain values for rainfall overall map...", log_run_id
#         )

#         # Precompute frequently accessed config values
#         interp_config = cp["interp"]
#         limits_config = cp["limits"]
#         time_config = cp["time"]
#         raingrids_config = cp["raingrids"]

#         # resample values to 1h means - using list comprehension is more efficient
#         calc_data_1h = xr.concat(
#             [cml.R.resample(time="1H", label="right").mean() for cml in calc_data],
#             dim="cml_id",
#         ).to_dataset()

#         logger.debug(
#             "[%s] Interpolating spatial data for rainfall overall map...", log_run_id
#         )

#         # Compute center coordinates once
#         calc_data_1h["lat_center"] = (
#             calc_data_1h.site_a_latitude + calc_data_1h.site_b_latitude
#         ) * 0.5  # multiplication is faster than division
#         calc_data_1h["lon_center"] = (
#             calc_data_1h.site_a_longitude + calc_data_1h.site_b_longitude
#         ) * 0.5

#         # Initialize interpolator once
#         interpolator = IdwKdtreeInterpolator(
#             nnear=interp_config["idw_near"],
#             p=interp_config["idw_power"],
#             exclude_nan=True,
#             max_distance=interp_config["idw_dist"],
#         )

#         # calculate coordinate grids with defined area boundaries
#         x_coords = np.arange(
#             limits_config["x_min"],
#             limits_config["x_max"],
#             interp_config["interp_res"],
#         )
#         y_coords = np.arange(
#             limits_config["y_min"],
#             limits_config["y_max"],
#             interp_config["interp_res"],
#         )
#         x_grid, y_grid = np.meshgrid(x_coords, y_coords)

#         # Compute mean across channels and sum across time more efficiently
#         rain_grid = interpolator(
#             x=calc_data_1h.lon_center,
#             y=calc_data_1h.lat_center,
#             z=calc_data_1h.R.mean(dim="channel_id").sum(dim="time"),
#             xgrid=x_grid,
#             ygrid=y_grid,
#         )

#         # Find min/max times more efficiently using numpy
#         all_times = np.concatenate([link.time.values for link in calc_data])
#         data_start, data_end = all_times.min(), all_times.max()

#         # *******************************************************************
#         # ***** SECOND PART: Calculate individual fields for animation ******
#         # *******************************************************************

#         # continue only if is it desired, else end
#         if not raingrids_config["is_only_overall"]:
#             logger.debug(
#                 "[%s] Resampling data for rainfall animation maps...", log_run_id
#             )

#             # Determine which resampling approach to use
#             output_step = time_config["output_step"]
#             time_step = time_config["step"]

#             if output_step == 60:  # if case of one hour steps, use existing resamples
#                 calc_data_steps = calc_data_1h
#             elif output_step > time_step:
#                 calc_data_steps = xr.concat(
#                     [
#                         cml.R.resample(time=f"{output_step}T", label="right").mean()
#                         for cml in calc_data
#                     ],
#                     dim="cml_id",
#                 ).to_dataset()
#             elif output_step == time_step:  # no resample needed
#                 calc_data_steps = xr.concat(calc_data, dim="cml_id")
#             else:
#                 raise ValueError("Invalid value of output_steps")

#             del calc_data  # Free memory early

#             # calculate totals instead of intensities, if desired
#             if raingrids_config["is_output_total"]:
#                 # 60 = 1 hour, since rain intensity is measured in mm/hour
#                 calc_data_steps["R"] = calc_data_steps.R / (60 / output_step)

#             logger.debug(
#                 "[%s] Interpolating spatial data for rainfall animation maps...",
#                 log_run_id,
#             )

#             # Compute center coordinates if not already done (output_step != 60)
#             if output_step != 60:
#                 calc_data_steps["lat_center"] = (
#                     calc_data_steps.site_a_latitude + calc_data_steps.site_b_latitude
#                 ) * 0.5
#                 calc_data_steps["lon_center"] = (
#                     calc_data_steps.site_a_longitude + calc_data_steps.site_b_longitude
#                 ) * 0.5

#             # Precompute frequently accessed values
#             min_rain_value = raingrids_config["min_rain_value"]
#             grids_to_del = realtime_runs - 1 if realtime_runs > 1 else 0

#             # interpolate each frame
#             for i, time_val in enumerate(calc_data_steps.time.values):
#                 grid = interpolator(
#                     x=calc_data_steps.lon_center,
#                     y=calc_data_steps.lat_center,
#                     z=calc_data_steps.R.mean(dim="channel_id").isel(time=i),
#                     xgrid=x_grid,
#                     ygrid=y_grid,
#                 )
#                 np.putmask(grid, grid < min_rain_value, 0)  # more efficient zeroing
#                 rain_grids.append(grid)
#                 last_time = time_val

#             # Remove old grids if in realtime mode
#             if grids_to_del > 0:
#                 del rain_grids[:grids_to_del]

#             return rain_grids, calc_data_steps, x_grid, y_grid, realtime_runs, last_time

#     except Exception as error:
#         logger.error(
#             "[%s] An error occurred during rainfall fields generation: %s %s.\n"
#             "Calculation thread terminated.",
#             log_run_id,
#             type(error).__name__,
#             str(error),
#         )
#         traceback.print_exc()
#         raise RainfieldsGenException(
#             "Error occurred during rainfall fields generation processing"
#         )


@measure_time
def generate_rainfields(
    calc_data: list[xr.Dataset],
    cp: dict[str, Any],
    rain_grids: list[np.ndarray],
    realtime_runs: int,
    last_time: np.datetime64,
    log_run_id: str = "default",
) -> tuple[list[np.ndarray], int, np.datetime64]:
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

        calc_data_1h["lat_center"] = (
            calc_data_1h.site_a_latitude + calc_data_1h.site_b_latitude
        ) / 2
        calc_data_1h["lon_center"] = (
            calc_data_1h.site_a_longitude + calc_data_1h.site_b_longitude
        ) / 2

        interpolator = IdwKdtreeInterpolator(
            nnear=cp["interp"]["idw_near"],
            p=cp["interp"]["idw_power"],
            exclude_nan=True,
            max_distance=cp["interp"]["idw_dist"],
        )

        # calculate coordinate grids with defined area boundaries
        x_coords = np.arange(
            cp["limits"]["x_min"], cp["limits"]["x_max"], cp["interp"]["interp_res"]
        )
        y_coords = np.arange(
            cp["limits"]["y_min"], cp["limits"]["y_max"], cp["interp"]["interp_res"]
        )
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        rain_grid = interpolator(
            x=calc_data_1h.lon_center,
            y=calc_data_1h.lat_center,
            z=calc_data_1h.R.mean(dim="channel_id").sum(dim="time"),
            xgrid=x_grid,
            ygrid=y_grid,
        )

        # get start and end timestamps from lists of DataArrays
        data_start = calc_data[0].time.min()
        data_end = calc_data[0].time.max()

        for link in calc_data:
            times = link.time.values
            data_start = min(data_start, times.min())
            data_end = max(data_end, times.max())

        # *******************************************************************
        # ***** SECOND PART: Calculate individual fields for animation ******
        # *******************************************************************

        # continue only if is it desired, else end
        if not cp["raingrids"]["is_only_overall"]:

            logger.debug(
                "[%s] Resampling data for rainfall animation maps...", log_run_id
            )

            # resample data to desired resolution, if needed
            if (
                cp["time"]["output_step"] == 60
            ):  # if case of one hour steps, use already existing resamples
                calc_data_steps = calc_data_1h
            elif cp["time"]["output_step"] > cp["time"]["step"]:
                os = cp["time"]["output_step"]
                calc_data_steps = xr.concat(
                    objs=[
                        cml.R.resample(time=f"{os}T", label="right").mean()
                        for cml in calc_data
                    ],
                    dim="cml_id",
                ).to_dataset()
            elif (
                cp["time"]["output_step"] == cp["time"]["step"]
            ):  # in case of same intervals, no resample needed
                calc_data_steps = xr.concat(calc_data, dim="cml_id")
            else:
                raise ValueError("Invalid value of output_steps")

            del calc_data

            # calculate totals instead of intensities, if desired
            if cp["raingrids"]["is_output_total"]:
                # get calc ratio
                time_ratio = (
                    60 / cp["time"]["output_step"]
                )  # 60 = 1 hour, since rain intensity is measured in mm/hour
                # overwrite values with totals per output step interval
                calc_data_steps["R"] = calc_data_steps.R / time_ratio

            logger.debug(
                "[%s] Interpolating spatial data for rainfall animation maps...",
                log_run_id,
            )

            # if output step is 60, it's already done
            if cp["time"]["output_step"] != 60:
                # central points of the links are considered in interpolation algorithms
                calc_data_steps["lat_center"] = (
                    calc_data_steps.site_a_latitude + calc_data_steps.site_b_latitude
                ) / 2
                calc_data_steps["lon_center"] = (
                    calc_data_steps.site_a_longitude + calc_data_steps.site_b_longitude
                ) / 2

            grids_to_del = 0
            # interpolate each frame
            for x in range(calc_data_steps.time.size):
                grid = interpolator(
                    x=calc_data_steps.lon_center,
                    y=calc_data_steps.lat_center,
                    z=calc_data_steps.R.mean(dim="channel_id").isel(time=x),
                    xgrid=x_grid,
                    ygrid=y_grid,
                )
                grid[grid < cp["raingrids"]["min_rain_value"]] = (
                    0  # zeroing out small values below threshold
                )
                rain_grids.append(grid)
                last_time = calc_data_steps.time[x].values

                if realtime_runs > 1:
                    grids_to_del += 1

            for x in range(grids_to_del):
                del rain_grids[x]

            # del calc_data_steps
            if calc_data_1h is not None:
                del calc_data_1h

            return rain_grids, calc_data_steps, x_grid, y_grid, realtime_runs, last_time

    except BaseException as error:
        logger.error(
            "[%s] An error occurred during rainfall fields generation: %s %s.\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
        )

        traceback.print_exc()

        raise RainfieldsGenException(
            "Error occurred during rainfall fields generation processing"
        )


@measure_time
def generate_rainfields_historic(
    calc_data: list[xr.Dataset],
    cp: dict[str, Any],
    rain_grids: list[np.ndarray],
    log_run_id: str = "default",
) -> tuple[list[np.ndarray], int, np.datetime64]:
    try:
        # **********************************************************************
        # ***** FIRST PART: Calculate overall rainfall accumulation field ******
        # **********************************************************************

        logger.info(
            "[%s] Resampling rain values for rainfall overall map...", log_run_id
        )

        # resample values to 1h means
        calc_data_1h = xr.concat(
            objs=[cml.R.resample(time="1H", label="right").mean() for cml in calc_data],
            dim="cml_id",
        ).to_dataset()

        logger.info(
            "[%s] Interpolating spatial data for rainfall overall map...", log_run_id
        )

        # TODO: use already created coords from external filter
        # if not cp['is_external_filter_enabled']:
        # central points of the links are considered in interpolation algorithms
        calc_data_1h["lat_center"] = (
            calc_data_1h.site_a_latitude + calc_data_1h.site_b_latitude
        ) / 2
        calc_data_1h["lon_center"] = (
            calc_data_1h.site_a_longitude + calc_data_1h.site_b_longitude
        ) / 2

        interpolator = IdwKdtreeInterpolator(
            nnear=cp["interp"]["idw_near"],
            p=cp["interp"]["idw_power"],
            exclude_nan=True,
            max_distance=cp["interp"]["idw_dist"],
        )

        # calculate coordinate grids with defined area boundaries
        x_coords = np.arange(
            cp["limits"]["x_min"], cp["limits"]["x_max"], cp["interp"]["interp_res"]
        )
        y_coords = np.arange(
            cp["limits"]["y_min"], cp["limits"]["y_max"], cp["interp"]["interp_res"]
        )
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        rain_grid = interpolator(
            x=calc_data_1h.lon_center,
            y=calc_data_1h.lat_center,
            z=calc_data_1h.R.mean(dim="channel_id").sum(dim="time"),
            xgrid=x_grid,
            ygrid=y_grid,
        )

        # get start and end timestamps from lists of DataArrays
        data_start = calc_data[0].time.min()
        data_end = calc_data[0].time.max()

        for link in calc_data:
            times = link.time.values
            data_start = min(data_start, times.min())
            data_end = max(data_end, times.max())

        # *******************************************************************
        # ***** SECOND PART: Calculate individual fields for animation ******
        # *******************************************************************

        # continue only if is it desired, else end
        if not cp["raingrids"]["is_only_overall"]:

            logger.info(
                "[%s] Resampling data for rainfall animation maps...", log_run_id
            )

            # resample data to desired resolution, if needed
            if (
                cp["time"]["output_step"] == 60
            ):  # if case of one hour steps, use already existing resamples
                calc_data_steps = calc_data_1h
            elif cp["time"]["output_step"] > cp["time"]["step"]:
                os = cp["time"]["output_step"]
                calc_data_steps = xr.concat(
                    objs=[
                        cml.R.resample(time=f"{os}T", label="right").mean()
                        for cml in calc_data
                    ],
                    dim="cml_id",
                ).to_dataset()
            elif (
                cp["time"]["output_step"] == cp["time"]["step"]
            ):  # in case of same intervals, no resample needed
                calc_data_steps = xr.concat(calc_data, dim="cml_id")
            else:
                raise ValueError("Invalid value of output_steps")

            del calc_data

            # calculate totals instead of intensities, if desired
            if cp["raingrids"]["is_output_total"]:
                # get calc ratio
                time_ratio = (
                    60 / cp["time"]["output_step"]
                )  # 60 = 1 hour, since rain intensity is measured in mm/hour
                # overwrite values with totals per output step interval
                calc_data_steps["R"] = calc_data_steps.R / time_ratio

            logger.info(
                "[%s] Interpolating spatial data for rainfall animation maps...",
                log_run_id,
            )

            # if output step is 60, it's already done
            if cp["time"]["output_step"] != 60:
                # central points of the links are considered in interpolation algorithms
                calc_data_steps["lat_center"] = (
                    calc_data_steps.site_a_latitude + calc_data_steps.site_b_latitude
                ) / 2
                calc_data_steps["lon_center"] = (
                    calc_data_steps.site_a_longitude + calc_data_steps.site_b_longitude
                ) / 2

            # interpolate each frame
            for x in range(calc_data_steps.time.size):
                grid = interpolator(
                    x=calc_data_steps.lon_center,
                    y=calc_data_steps.lat_center,
                    z=calc_data_steps.R.mean(dim="channel_id").isel(time=x),
                    xgrid=x_grid,
                    ygrid=y_grid,
                )
                grid[grid < cp["raingrids"]["min_rain_value"]] = (
                    0  # zeroing out small values below threshold
                )
                rain_grids.append(grid)

            # del calc_data_steps
            if calc_data_1h is not None:
                del calc_data_1h

            return rain_grids, calc_data_steps, x_grid, y_grid

    except BaseException as error:
        logger.error(
            "[%s] An error occurred during rainfall fields generation: %s %s.\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
        )

        traceback.print_exc()

        raise RainfieldsGenException(
            "Error occurred during rainfall fields generation processing"
        )
