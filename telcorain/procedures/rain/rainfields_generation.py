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


def _to_int(val, default):
    return int(round(_to_float(val, default)))


def _interpolate_grid_chunked(
    interpolator,
    *,
    x_sites,
    y_sites,
    z_values,
    x_grid,
    y_grid,
    chunk_rows: int,
):
    if chunk_rows <= 0 or chunk_rows >= x_grid.shape[0]:
        return interpolator(x=x_sites, y=y_sites, z=z_values, xgrid=x_grid, ygrid=y_grid)

    out = np.full(x_grid.shape, np.nan, dtype=float)
    for row_start in range(0, x_grid.shape[0], chunk_rows):
        row_end = min(x_grid.shape[0], row_start + chunk_rows)
        out[row_start:row_end, :] = interpolator(
            x=x_sites,
            y=y_sites,
            z=z_values,
            xgrid=x_grid[row_start:row_end, :],
            ygrid=y_grid[row_start:row_end, :],
        )
    return out


@measure_time
def generate_rainfields(
    calc_data: list[xr.Dataset],
    config: dict[str, Any],
    rain_grids: list[np.ndarray],
    rain_grids_sum: Optional[list[np.ndarray]] = None,
    *,
    is_historic: bool = False,
    realtime_runs: int = 1,
    last_time: Optional[np.datetime64] = None,
    log_run_id: str = "default",
    historic_flush_every: int = 0,
    historic_flush_callback=None,
):
    """
    Generates spatial rainfields for:
      - normal intensity grids (mm/h), always
      - hour-sum grids (mm), only when config["hour_sum"]["enabled"] is True

      [setting].dry_as_nan = true
        - if true, link value == 0 is converted to NaN before interpolation
        - interpolator has exclude_nan=True so those links are ignored (telcorain synth version behavior)
        - this is applied for both mm/h and hour-sum mm series

    Realtime behavior:
      - uses the whole window as context
      - interpolates and returns only newly available timesteps
    """
    if rain_grids_sum is None:
        rain_grids_sum = []

    try:
        logger.info("[%s] Generating rainfields...", log_run_id)

        if not calc_data:
            logger.warning("[%s] Empty calc_data, nothing to interpolate.", log_run_id)
            if is_historic:
                return rain_grids, rain_grids_sum, None, None, None
            return (
                rain_grids,
                rain_grids_sum,
                None,
                None,
                None,
                realtime_runs,
                last_time,
            )

        # --------------------------------------------------------------
        # 0) Concatenate all links once and precompute geometry
        # --------------------------------------------------------------
        ds_all = xr.concat(calc_data, dim="cml_id")

        interp_cfg = config["interp"]
        limits = config["limits"]

        use_mercator = bool(interp_cfg.get("use_mercator", False))
        transformer: Optional[Transformer] = None
        if use_mercator:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        lat_center = (ds_all.site_a_latitude + ds_all.site_b_latitude) / 2
        lon_center = (ds_all.site_a_longitude + ds_all.site_b_longitude) / 2

        if use_mercator:
            x_sites, y_sites = transformer.transform(
                lon_center.values.astype(float),
                lat_center.values.astype(float),
            )
        else:
            x_sites = lon_center.values.astype(float)
            y_sites = lat_center.values.astype(float)

        ds_all = ds_all.assign(x_center=("cml_id", x_sites))
        ds_all = ds_all.assign(y_center=("cml_id", y_sites))

        # --------------------------------------------------------------
        # 1) Create IDW interpolator & target grid
        # --------------------------------------------------------------
        if use_mercator:
            x_min_deg = float(limits["x_min"])
            x_max_deg = float(limits["x_max"])
            y_min_deg = float(limits["y_min"])
            y_max_deg = float(limits["y_max"])

            x_min_m, y_min_m = transformer.transform(x_min_deg, y_min_deg)
            x_max_m, y_max_m = transformer.transform(x_max_deg, y_max_deg)

            x_lo, x_hi = sorted([x_min_m, x_max_m])
            y_lo, y_hi = sorted([y_min_m, y_max_m])

            step_raw = interp_cfg.get("grid_step_m", None)
            nx_cfg = interp_cfg.get("grid_nx", None)
            ny_cfg = interp_cfg.get("grid_ny", None)

            if step_raw not in [None, ""]:
                step_m = _to_float(step_raw, 1000.0)
                width_m = x_hi - x_lo
                height_m = y_hi - y_lo
                nx = max(1, int(np.ceil(width_m / step_m)))
                ny = max(1, int(np.ceil(height_m / step_m)))

                # Keep truly square cells in projected space and center any remainder
                # as symmetric padding around the requested bbox.
                dx = step_m
                dy = step_m
                x_pad = max(0.0, nx * step_m - width_m) / 2.0
                y_pad = max(0.0, ny * step_m - height_m) / 2.0
                x_origin = x_lo - x_pad
                y_origin = y_lo - y_pad
                x_coords = x_origin + (np.arange(nx) + 0.5) * dx
                y_coords = y_origin + (np.arange(ny) + 0.5) * dy
                logger.debug(
                    "[%s] Using exact Mercator grid_step_m=%.2f m -> nx=%d, ny=%d",
                    log_run_id,
                    step_m,
                    nx,
                    ny,
                )
                logger.debug(
                    "[%s] Step-grid padded extent: x_origin=%.1f, y_origin=%.1f, width=%.1f m, height=%.1f m",
                    log_run_id,
                    x_origin,
                    y_origin,
                    nx * step_m,
                    ny * step_m,
                )
            elif nx_cfg is not None and ny_cfg is not None:
                nx = int(nx_cfg)
                ny = int(ny_cfg)
                dx = (x_hi - x_lo) / nx
                dy = (y_hi - y_lo) / ny
                x_coords = x_lo + (np.arange(nx) + 0.5) * dx
                y_coords = y_lo + (np.arange(ny) + 0.5) * dy
                logger.debug(
                    "[%s] Using explicit Mercator grid: nx=%d, ny=%d, dx=%.2f m, dy=%.2f m",
                    log_run_id,
                    nx,
                    ny,
                    dx,
                    dy,
                )
            else:
                step_m = 1000.0
                width_m = x_hi - x_lo
                height_m = y_hi - y_lo
                nx = max(1, int(np.ceil(width_m / step_m)))
                ny = max(1, int(np.ceil(height_m / step_m)))
                dx = step_m
                dy = step_m
                x_pad = max(0.0, nx * step_m - width_m) / 2.0
                y_pad = max(0.0, ny * step_m - height_m) / 2.0
                x_origin = x_lo - x_pad
                y_origin = y_lo - y_pad
                x_coords = x_origin + (np.arange(nx) + 0.5) * dx
                y_coords = y_origin + (np.arange(ny) + 0.5) * dy
                logger.debug(
                    "[%s] Falling back to exact Mercator grid_step_m=%.2f m -> nx=%d, ny=%d",
                    log_run_id,
                    step_m,
                    nx,
                    ny,
                )

            logger.debug(
                "[%s] Limits (deg): x_min=%.6f, x_max=%.6f, y_min=%.6f, y_max=%.6f",
                log_run_id,
                x_min_deg,
                x_max_deg,
                y_min_deg,
                y_max_deg,
            )
            logger.debug(
                "[%s] Mercator extent: x_lo=%.1f, x_hi=%.1f (Î”x=%.1f m), "
                "y_lo=%.1f, y_hi=%.1f (Î”y=%.1f m)",
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

        x_sites = ds_all.x_center.values
        y_sites = ds_all.y_center.values

        chunk_rows_raw = interp_cfg.get("idw_chunk_rows", None)
        chunk_rows = _to_int(chunk_rows_raw, 0) if chunk_rows_raw not in [None, ""] else 0

        nnear = int(interp_cfg["idw_near"])
        p = _to_float(interp_cfg["idw_power"], interp_cfg["idw_power"])

        if use_mercator:
            max_distance = _to_float(interp_cfg.get("idw_dist_m", 20000.0), 20000.0)
        else:
            max_distance = _to_float(interp_cfg.get("idw_dist", 0.4), 0.4)

        interpolator = IdwKdtreeInterpolator(
            nnear=nnear,
            p=p,
            exclude_nan=True,
            max_distance=max_distance,
        )

        # --------------------------------------------------------------
        # 2) Time-step rainfall fields for animation
        # --------------------------------------------------------------
        calc_data_1h = ds_all.R.resample(time="1H", label="right").mean().to_dataset()

        ts = int(config["time"]["output_step"])  # minutes
        base_step = int(config["time"]["step"])  # minutes

        if ts == 60:
            calc_data_steps = calc_data_1h
        elif ts > base_step:
            calc_data_steps = (
                ds_all.R.resample(time=f"{ts}T", label="right").mean().to_dataset()
            )
        elif ts == base_step:
            calc_data_steps = ds_all
        else:
            raise ValueError("Invalid value of output_step")

        logger.debug(
            "[%s] Interpolating spatial data for rainfall animation maps...", log_run_id
        )

        z_all = calc_data_steps.R.mean(dim="channel_id").values  # (cml_id, time)
        times = calc_data_steps.time.values
        min_rain = float(config["raingrids"]["min_rain_value"])

        # synth-like behavior: only 0 becomes NaN, no thresholding on links
        dry_as_nan = config["setting"]["dry_as_nan"]

        # --------------------------------------------------------------
        # 2b) Hour-sum computation on link-series (before spatial interpolation)
        # --------------------------------------------------------------
        hs_cfg = config.get("hour_sum", {})
        hour_sum_enabled = bool(hs_cfg.get("enabled", False))
        hour_sum_win_min = int(hs_cfg.get("window_minutes", 60))

        window_hours = float(hour_sum_win_min) / 60.0
        min_rain_mm = min_rain * window_hours

        if hour_sum_enabled:
            dt_hours = float(ts) / 60.0
            z_step_mm = z_all * dt_hours  # mm/h -> mm per step

            win_steps = int(round(hour_sum_win_min / ts))
            if win_steps < 1:
                win_steps = 1

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

            calc_data_steps["R_hour_sum"] = (("cml_id", "time"), z_hour_sum_all)
        else:
            z_hour_sum_all = None

        # --------------------------------------------------------------
        # 2c) Determine which timesteps are new in realtime mode
        # --------------------------------------------------------------
        T = int(z_all.shape[1])
        if is_historic:
            time_indices = list(range(T))
            calc_data_out = calc_data_steps
            out_rain_grids = rain_grids
            out_rain_grids_sum = rain_grids_sum
        else:
            if last_time is None:
                new_mask = np.ones(T, dtype=bool)
            else:
                new_mask = times > last_time

            time_indices = np.flatnonzero(new_mask).tolist()
            calc_data_out = calc_data_steps.isel(time=time_indices)
            out_rain_grids = []
            out_rain_grids_sum = []

            if not time_indices:
                logger.info(
                    "[%s] No new rainfield timesteps detected after %s.",
                    log_run_id,
                    last_time,
                )
                return (
                    out_rain_grids,
                    out_rain_grids_sum,
                    calc_data_out,
                    x_grid,
                    y_grid,
                    realtime_runs,
                    last_time,
                )

        stream_historic = bool(
            is_historic and historic_flush_callback is not None and historic_flush_every > 0
        )
        chunk_time_indices: list[int] = []
        interp_chunk_rows = chunk_rows if stream_historic else 0
        if stream_historic:
            logger.info(
                "[%s] Streaming historic outputs every %d timesteps to reduce memory usage.",
                log_run_id,
                historic_flush_every,
            )
            if interp_chunk_rows > 0 and interp_chunk_rows < x_grid.shape[0]:
                logger.info(
                    "[%s] Using chunked interpolation with %d row chunks for grid %dx%d.",
                    log_run_id,
                    interp_chunk_rows,
                    x_grid.shape[1],
                    x_grid.shape[0],
                )

        # --------------------------------------------------------------
        # 2d) Spatial interpolation only for selected timesteps
        # --------------------------------------------------------------
        total_steps = len(time_indices)
        for step_idx, i in enumerate(time_indices, start=1):
            if stream_historic and (step_idx == 1 or step_idx == total_steps or step_idx % 50 == 0):
                logger.info(
                    "[%s] Interpolating timestep %d/%d.",
                    log_run_id,
                    step_idx,
                    total_steps,
                )

            # ---- intensity mm/h ----
            z_t = np.asarray(z_all[:, i], dtype=float).copy()

            if dry_as_nan:
                z_t[z_t == 0.0] = np.nan

            grid = _interpolate_grid_chunked(
                interpolator,
                x_sites=x_sites,
                y_sites=y_sites,
                z_values=z_t,
                x_grid=x_grid,
                y_grid=y_grid,
                chunk_rows=interp_chunk_rows,
            )

            grid[grid < min_rain] = 0.0
            out_rain_grids.append(grid)

            # ---- hour-sum mm ----
            if hour_sum_enabled and z_hour_sum_all is not None:
                zsum_t = z_hour_sum_all[:, i]
                grid_sum = _interpolate_grid_chunked(
                    interpolator,
                    x_sites=x_sites,
                    y_sites=y_sites,
                    z_values=zsum_t,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    chunk_rows=interp_chunk_rows,
                )

                # keep NaNs transparent
                grid_sum[~np.isfinite(grid_sum)] = np.nan

                # remove tiny IDW halos by applying the mm-equivalent of min_rain
                grid_sum[grid_sum < min_rain_mm] = 0.0

                out_rain_grids_sum.append(grid_sum)

            if stream_historic:
                chunk_time_indices.append(i)
                if len(chunk_time_indices) >= historic_flush_every:
                    historic_flush_callback(
                        rain_grids=out_rain_grids,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        calc_dataset=calc_data_out.isel(time=chunk_time_indices),
                        rain_grids_sum=(out_rain_grids_sum if hour_sum_enabled else None),
                    )
                    out_rain_grids = []
                    out_rain_grids_sum = []
                    chunk_time_indices = []

            if not is_historic:
                last_time = times[i]

        if stream_historic and chunk_time_indices:
            historic_flush_callback(
                rain_grids=out_rain_grids,
                x_grid=x_grid,
                y_grid=y_grid,
                calc_dataset=calc_data_out.isel(time=chunk_time_indices),
                rain_grids_sum=(out_rain_grids_sum if hour_sum_enabled else None),
            )
            out_rain_grids = []
            out_rain_grids_sum = []
            chunk_time_indices = []

        # --------------------------------------------------------------
        # 3) Return
        # --------------------------------------------------------------
        if is_historic:
            if stream_historic:
                return out_rain_grids, out_rain_grids_sum, calc_data_out.isel(time=slice(0, 0)), x_grid, y_grid
            return out_rain_grids, out_rain_grids_sum, calc_data_out, x_grid, y_grid

        return (
            out_rain_grids,
            out_rain_grids_sum,
            calc_data_out,
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


