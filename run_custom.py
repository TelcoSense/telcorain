import argparse
import json
import warnings
from datetime import datetime

warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from telcorain.calculation import Calculation
from pathlib import Path

from telcorain.custom_data import (
    export_custom_cml_metadata_json,
    export_custom_rain_event_summary_csv,
    export_openrainer_reference_pngs,
    is_influx_source,
)
from telcorain.database.influx_manager import influx_man
from telcorain.handlers import logger
from telcorain.helpers import create_config_dict, ensure_utc
from telcorain.writer import Writer

warnings.simplefilter(action="ignore", category=FutureWarning)


def deep_merge_config(base: dict, updates: dict) -> dict:
    result = base.copy()
    for key, val in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge_config(result[key], val)
        else:
            result[key] = val
    return result


def _normalize_custom_cfg_payload(cfg: dict) -> dict:
    cfg = cfg.copy()
    if "time" in cfg and isinstance(cfg["time"], dict):
        time_cfg = cfg["time"].copy()
        for key in ("start", "end"):
            if key in time_cfg and isinstance(time_cfg[key], str):
                time_cfg[key] = datetime.fromisoformat(time_cfg[key].replace("Z", "+00:00"))
            if key in time_cfg and isinstance(time_cfg[key], datetime):
                time_cfg[key] = ensure_utc(time_cfg[key])
        cfg["time"] = time_cfg
    return cfg


def run_custom(config_path: str = "configs/config.ini", cfg_updates: dict | None = None) -> int:
    config = create_config_dict(path=config_path, format=True)
    if cfg_updates:
        config = deep_merge_config(config, _normalize_custom_cfg_payload(cfg_updates))
    if is_influx_source(config):
        logger.error(
            "run_custom.py is intended for non-influx custom datasets. "
            "Set [data_source] mode to pycomlink_example, pycomlink_netcdf, netherlands_raw_csv, or openrainer_tar."
        )
        return 2

    start_time = datetime.now()
    logger.info("Starting custom dataset calculation at: %s", start_time)

    writer = Writer(
        influx_man=influx_man,
        write_influx_intensities=config["setting"]["write_influx_intensities"],
        config=config,
        is_historic=False,
    )

    source_cfg = config.get("data_source", {})
    if bool(source_cfg.get("export_rain_event_summary_csv", False)):
        summary_name = str(source_cfg.get("rain_event_summary_filename", "rain_event_summary.csv")).strip() or "rain_event_summary.csv"
        summary_path = Path(str(config["directories"]["outputs_json"])) / summary_name
        if summary_path.exists():
            summary_path.unlink()

    flush_every = int(config.get("historic", {}).get("flush_every_timesteps", 100))

    def _flush_chunk(*, rain_grids, x_grid, y_grid, calc_dataset, rain_grids_sum=None):
        export_custom_rain_event_summary_csv(
            config=config,
            calc_dataset=calc_dataset,
            log_run_id="Custom run",
        )
        writer.push_results(
            rain_grids=rain_grids,
            x_grid=x_grid,
            y_grid=y_grid,
            calc_dataset=calc_dataset,
            rain_grids_sum=rain_grids_sum,
        )

    calculation = Calculation(
        influx_man=influx_man,
        links={},
        selection={},
        config=config,
        is_historic=True,
        results_id=0,
    )
    calculation.run(
        historic_flush_every=flush_every,
        historic_flush_callback=_flush_chunk,
    )

    export_custom_cml_metadata_json(
        config=config,
        links=calculation.links,
        calc_data=calculation.calc_data_steps,
        log_run_id="Custom run",
    )
    export_openrainer_reference_pngs(
        config=config,
        log_run_id="Custom run",
    )

    if not calculation.results_streamed:
        export_custom_rain_event_summary_csv(
            config=config,
            calc_dataset=calculation.calc_data_steps,
            log_run_id="Custom run",
        )
        writer.push_results(
            rain_grids=calculation.rain_grids,
            x_grid=calculation.x_grid,
            y_grid=calculation.y_grid,
            calc_dataset=calculation.calc_data_steps,
            rain_grids_sum=calculation.rain_grids_sum,
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.ini", help="Path to config.ini")
    parser.add_argument("--cfg", help="JSON string with config overrides (same style as run_web.py)")
    args = parser.parse_args()
    cfg_updates = json.loads(args.cfg) if args.cfg else None
    raise SystemExit(run_custom(args.config, cfg_updates=cfg_updates))
