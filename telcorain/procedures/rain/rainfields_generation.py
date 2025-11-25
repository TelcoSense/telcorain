import traceback
from typing import Any, Optional

import numpy as np
import xarray as xr
from pycomlink.spatial.interpolator import IdwKdtreeInterpolator
from pyproj import Transformer

from telcorain.handlers import logger
from telcorain.procedures.exceptions import RainfieldsGenException
from telcorain.helpers import measure_time


def _to_float(val, default):
    """Convert config value to float, stripping inline comments."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # remove inline comments starting with ';' or '#'
        for sep in (";", "#"):
            val = val.split(sep, 1)[0]
        val = val.strip()
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    # fall back if parsing fails
    logger.warning(
        "Could not parse float from config value %r, using default %s",
        val,
        default,
    )
    return float(default)


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

        interp_cfg = config["interp"]
        limits = config["limits"]

        # Use Mercator or plain lon/lat?
        use_mercator = bool(interp_cfg.get("use_mercator", False))
        transformer: Optional[Transformer] = None
        if use_mercator:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # ------------------------------------------------------------------
        # 1) Create IDW interpolator & target grid (only once)
        # ------------------------------------------------------------------
        # Grid coordinates
        if use_mercator:
            # limits are in lon/lat (deg) -> transform to metres
            x_min_deg = float(limits["x_min"])
            x_max_deg = float(limits["x_max"])
            y_min_deg = float(limits["y_min"])
            y_max_deg = float(limits["y_max"])

            x_min_m, y_min_m = transformer.transform(x_min_deg, y_min_deg)
            x_max_m, y_max_m = transformer.transform(x_max_deg, y_max_deg)

            x_lo, x_hi = sorted([x_min_m, x_max_m])
            y_lo, y_hi = sorted([y_min_m, y_max_m])

            step_m = float(interp_cfg.get("grid_step_m", 1000.0))  # 1 km default
            x_coords = np.arange(x_lo, x_hi, step_m)
            y_coords = np.arange(y_lo, y_hi, step_m)
        else:
            # Original behaviour: lon/lat grid in degrees
            x_coords = np.arange(
                limits["x_min"],
                limits["x_max"],
                interp_cfg["interp_res"],
            )
            y_coords = np.arange(
                limits["y_min"],
                limits["y_max"],
                interp_cfg["interp_res"],
            )

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Site coordinates in the same CRS as the grid
        if use_mercator:
            lons = ds_all.lon_center.values.astype(float)
            lats = ds_all.lat_center.values.astype(float)
            x_sites, y_sites = transformer.transform(lons, lats)
        else:
            x_sites = ds_all.lon_center.values
            y_sites = ds_all.lat_center.values

        # IDW parameters
        nnear = interp_cfg["idw_near"]
        p = interp_cfg["idw_power"]

        if use_mercator:
            max_distance = _to_float(
                interp_cfg.get("idw_dist_m", 20000.0), 20000.0
            )  # metres
        else:
            max_distance = interp_cfg["idw_dist"]  # degrees, original

        interpolator = IdwKdtreeInterpolator(
            nnear=nnear,
            p=p,
            exclude_nan=True,
            max_distance=max_distance,
        )

        # 1h resample from concatenated dataset
        calc_data_1h = ds_all.R.resample(time="1H", label="right").mean().to_dataset()

        # ------------------------------------------------------------------
        # 2) Time-step rainfall fields for animation
        # ------------------------------------------------------------------
        if not config["raingrids"]["is_only_overall"]:
            ts = config["time"]["output_step"]  # [minutes]
            base_step = config["time"]["step"]  # [minutes]

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
                time_ratio = 60.0 / float(ts)  # hours per step (kept as in original)
                calc_data_steps["R"] = calc_data_steps.R / time_ratio

            logger.debug(
                "[%s] Interpolating spatial data for rainfall animation maps...",
                log_run_id,
            )

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
            # 3) Return in the original shapes
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
