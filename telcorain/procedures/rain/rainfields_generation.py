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

        interp_cfg = config["interp"]
        limits = config["limits"]

        # Use Mercator or plain lon/lat?
        use_mercator = bool(interp_cfg.get("use_mercator", False))
        transformer: Optional[Transformer] = None
        if use_mercator:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Compute geographic centers (deg)
        lat_center = (ds_all.site_a_latitude + ds_all.site_b_latitude) / 2
        lon_center = (ds_all.site_a_longitude + ds_all.site_b_longitude) / 2

        # Convert all link centers to grid CRS *before interpolation*
        if use_mercator:
            x_sites, y_sites = transformer.transform(
                lon_center.values.astype(float),
                lat_center.values.astype(float),
            )
        else:
            x_sites = lon_center.values.astype(float)
            y_sites = lat_center.values.astype(float)

        # Attach back to dataset to keep compatibility if something needs it
        ds_all = ds_all.assign(x_center=("cml_id", x_sites))
        ds_all = ds_all.assign(y_center=("cml_id", y_sites))

        # ------------------------------------------------------------------
        # 1) Create IDW interpolator & target grid
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

            # Grid size to match CHMI
            nx_cfg = interp_cfg.get("grid_nx", None)
            ny_cfg = interp_cfg.get("grid_ny", None)

            if nx_cfg is not None and ny_cfg is not None:
                nx = int(nx_cfg)
                ny = int(ny_cfg)
                dx = (x_hi - x_lo) / nx
                dy = (y_hi - y_lo) / ny
                logger.debug(
                    "[%s] Using explicit Mercator grid: nx=%d, ny=%d, dx=%.2f m, dy=%.2f m",
                    log_run_id,
                    nx,
                    ny,
                    dx,
                    dy,
                )
            else:
                # Fallback: use grid_step_m (approximate 1 km)
                step_m = _to_float(interp_cfg.get("grid_step_m", 1000.0), 1000.0)
                nx = int(np.round((x_hi - x_lo) / step_m))
                ny = int(np.round((y_hi - y_lo) / step_m))

                # Recompute dx, dy from integer pixel counts
                dx = (x_hi - x_lo) / nx
                dy = (y_hi - y_lo) / ny
                logger.debug(
                    "[%s] Using step_m ~ %.2f m -> nx=%d, ny=%d, dx=%.2f m, dy=%.2f m",
                    log_run_id,
                    step_m,
                    nx,
                    ny,
                    dx,
                    dy,
                )

            # Pixel centers
            x_coords = x_lo + (np.arange(nx) + 0.5) * dx
            y_coords = y_lo + (np.arange(ny) + 0.5) * dy

            logger.debug(
                "[%s] Limits (deg): x_min=%.6f, x_max=%.6f, y_min=%.6f, y_max=%.6f",
                log_run_id,
                x_min_deg,
                x_max_deg,
                y_min_deg,
                y_max_deg,
            )
            logger.debug(
                "[%s] Mercator extent: x_lo=%.1f, x_hi=%.1f (Δx=%.1f m), "
                "y_lo=%.1f, y_hi=%.1f (Δy=%.1f m)",
                log_run_id,
                x_lo,
                x_hi,
                (x_hi - x_lo),
                y_lo,
                y_hi,
                (y_hi - y_lo),
            )
            logger.debug(
                "[%s] Grid shape: ny=%d, nx=%d; dx=%.2f m, dy=%.2f m",
                log_run_id,
                ny,
                nx,
                dx,
                dy,
            )

        else:
            # Or use lon/lat grid in degrees
            x_coords = np.arange(
                _to_float(limits["x_min"], limits["x_min"]),
                _to_float(limits["x_max"], limits["x_max"]),
                _to_float(interp_cfg["interp_res"], interp_cfg["interp_res"]),
            )
            y_coords = np.arange(
                _to_float(limits["y_min"], limits["y_min"]),
                _to_float(limits["y_max"], limits["y_max"]),
                _to_float(interp_cfg["interp_res"], interp_cfg["interp_res"]),
            )

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Site coordinates in the same CRS as the grid
        x_sites = ds_all.x_center.values
        y_sites = ds_all.y_center.values

        # IDW parameters
        nnear = int(interp_cfg["idw_near"])
        p = _to_float(interp_cfg["idw_power"], interp_cfg["idw_power"])

        if use_mercator:
            max_distance = _to_float(
                interp_cfg.get("idw_dist_m", 20000.0), 20000.0
            )  # metres
        else:
            max_distance = _to_float(interp_cfg.get("idw_dist", 0.4), 0.4)  # degrees

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

        logger.debug(
            "[%s] Interpolating spatial data for rainfall animation maps...",
            log_run_id,
        )

        # Precompute z(t) as NumPy for fast looping
        # shape: (cml_id, time)
        z_all = calc_data_steps.R.mean(dim="channel_id").values
        times = calc_data_steps.time.values
        min_rain = config["raingrids"]["min_rain_value"]

        # prepare hour sum computation
        rain_grids_sum: list[np.ndarray] = []
        hs_cfg = config.get("hour_sum", {})
        hour_sum_enabled = bool(hs_cfg.get("enabled", False))
        hour_sum_win_min = int(hs_cfg.get("window_minutes", 60))

        if hour_sum_enabled:
            dt_hours = ts / 60.0
            # convert mm/h -> mm per step
            z_step_mm = z_all * dt_hours

            win_steps = int(round(hour_sum_win_min / ts))
            if win_steps < 1:
                win_steps = 1

            # rolling sum over time axis (axis=1), strict full window
            z_hour_sum_all = np.full_like(z_step_mm, np.nan, dtype=float)
            logger.debug(
                "[%s] hour_sum: ts=%d min, win_steps=%d, first_valid_index=%d",
                log_run_id,
                ts,
                win_steps,
                win_steps - 1,
            )
            for j in range(win_steps - 1, z_step_mm.shape[1]):
                z_hour_sum_all[:, j] = np.nansum(
                    z_step_mm[:, j - win_steps + 1 : j + 1], axis=1
                )

            # attach into dataset (dims must match calc_data_steps: cml_id, time)
            calc_data_steps["R_hour_sum"] = (("cml_id", "time"), z_hour_sum_all)
        else:
            z_hour_sum_all = None

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

            # hour sum computation
            if hour_sum_enabled and z_hour_sum_all is not None:
                zsum_t = z_hour_sum_all[:, i]
                grid_sum = interpolator(
                    x=x_sites,
                    y=y_sites,
                    z=zsum_t,
                    xgrid=x_grid,
                    ygrid=y_grid,
                )
                # zero small values
                grid_sum[~np.isfinite(grid_sum)] = np.nan
                grid_sum[grid_sum < 0.0] = 0.0
                rain_grids_sum.append(grid_sum)

            if not is_historic:
                last_time = times[i]
                if realtime_runs > 1:
                    grids_to_del += 1

        # maintain sliding window
        if not is_historic and grids_to_del > 0:
            del rain_grids[:grids_to_del]
            if rain_grids_sum is not None:
                del rain_grids_sum[:grids_to_del]

        # ------------------------------------------------------------------
        # 3) Return in the original shapes
        # ------------------------------------------------------------------
        if is_historic:
            return rain_grids, rain_grids_sum, calc_data_steps, x_grid, y_grid
        else:
            return (
                rain_grids,
                rain_grids_sum,
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
