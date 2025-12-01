import warnings
from datetime import datetime

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers import logger
from telcorain.writer import Writer
from telcorain.calculation import Calculation
from telcorain.helpers import (
    create_config_dict,
    select_all_links,
    ensure_utc,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def deep_merge_config(base: dict, updates: dict) -> dict:
    result = base.copy()
    for key, val in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge_config(result[key], val)
        else:
            result[key] = val
    return result


def run_hist_calc(cfg: dict):
    preconfig = create_config_dict(path="./configs/config.ini", format=True)
    config = deep_merge_config(preconfig, cfg)
    config["time"]["start"] = ensure_utc(config["time"]["start"])
    config["time"]["end"] = ensure_utc(config["time"]["end"])

    start_time = datetime.now()
    logger.info("Starting the historic calculation at: %s", start_time)

    sql_man = SqlManager()
    links = sql_man.load_metadata(
        ids=config["user_info"]["links_id"],
        min_length=config["cml"]["min_length"],
        max_length=config["cml"]["max_length"],
        exclude_ids=config["cml"]["exclude_cmls"],
    )

    selected_links = select_all_links(links=links)
    calculation = Calculation(
        influx_man=influx_man,
        links=links,
        selection=selected_links,
        config=config,
        is_historic=True,
        results_id=0,
    )

    # run the calculation
    calculation.run()

    writer = Writer(
        influx_man=influx_man,
        skip_influx=config["realtime"]["skip_influx_write"],
        config=config,
        is_historic=True,
    )

    writer.push_results(
        rain_grids=calculation.rain_grids,
        x_grid=calculation.x_grid,
        y_grid=calculation.y_grid,
        calc_dataset=calculation.calc_data_steps,
    )


if __name__ == "__main__":
    # this comes from the web app settings
    cfg = {
        # time setting
        "time": {
            "step": 10,
            "output_step": 10,
            "start": datetime(2023, 10, 13, 6, 30, tzinfo=None),
            "end": datetime(2023, 10, 13, 10, 30, tzinfo=None),
        },
        # CML filtering
        "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": False},
        # user info for folder names and link selection (list of IDs)
        "user_info": {
            "folder_name": "kraken",
            "links_id": [i for i in range(1, 1000)],
        },
        "wet_dry": {
            "is_mlp_enabled": False,
            "rolling_hours": 1.0,
            "rolling_values": 10,
            "wet_dry_deviation": 0.8,
            "baseline_samples": 5,
        },
        "interp": {
            "idw_power": 2,
            "idw_near": 8,
            "idw_dist_m": 30000.0,
        },
        "rendering": {
            "is_crop_enabled": True,
        },
    }

    run_hist_calc(cfg)
