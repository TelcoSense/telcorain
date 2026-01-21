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
from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers import logger
from telcorain.helpers import create_config_dict, ensure_utc, select_all_links
from telcorain.writer_modified import Writer

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
        write_influx_intensities=config["setting"]["write_influx_intensities"],
        config=config,
        is_web=True,
    )

    writer.push_results(
        rain_grids=calculation.rain_grids,
        x_grid=calculation.x_grid,
        y_grid=calculation.y_grid,
        calc_dataset=calculation.calc_data_steps,
        rain_grids_sum=calculation.rain_grids_sum,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # this comes from the web app
    parser.add_argument("--cfg", required=True, help="JSON string with calc params")
    args = parser.parse_args()
    # load the settings from web app as json
    cfg = json.loads(args.cfg)
    # replace start and end times with python datetimes
    cfg["time"]["start"] = datetime.fromisoformat(
        cfg["time"]["start"].replace("Z", "+00:00")
    )
    cfg["time"]["end"] = datetime.fromisoformat(
        cfg["time"]["end"].replace("Z", "+00:00")
    )
    run_hist_calc(cfg)
