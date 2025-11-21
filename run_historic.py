import warnings
from datetime import datetime, timezone

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers import setup_init_logging, logger
from telcorain.writer import Writer
from telcorain.calculation import Calculation
from telcorain.helpers import (
    create_config_dict,
    select_all_links,
    ensure_utc,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
setup_init_logging(logger)


def deep_merge_config(base: dict, updates: dict) -> dict:
    result = base.copy()
    for key, val in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge_config(result[key], val)
        else:
            result[key] = val
    return result


def run_hist_calc(cfg: dict):
    # load global config dict
    preconfig = create_config_dict(path="./configs/config.ini", format=False)
    config = deep_merge_config(preconfig, cfg)
    config["time"]["start"] = ensure_utc(config["time"]["start"])
    config["time"]["end"] = ensure_utc(config["time"]["end"])

    start_time = datetime.now()
    logger.info("Starting the historic calculation at: %s", start_time)

    # create sql manager and filter out short links
    sql_man = SqlManager()
    # load link definitions from MariaDB
    links = sql_man.load_metadata(
        ids=config["user_info"]["links_id"],
        min_length=config["cml"]["min_length"],
        max_length=config["cml"]["max_length"],
        exclude_ids=True,
    )

    # select all links
    selected_links = select_all_links(links=links)
    # define calculation object
    calculation = Calculation(
        influx_man=influx_man,
        links=links,
        selection=selected_links,
        config=config,
        is_historic=True,
        results_id=0,
        compensate_historic=config["historic"]["compensate_historic"],
    )

    # run the calculation
    calculation.run()

    # create the writer object and write the results to disk
    writer = Writer(
        sql_man=sql_man,
        influx_man=influx_man,
        skip_influx=config["historic"]["skip_influx"],
        skip_sql=config["historic"]["skip_mariadb"],
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
        # time setting (probably dont change step and output_step)
        "time": {
            "step": 10,
            "output_step": 10,
            "start": datetime(2023, 10, 20, 3, 30, tzinfo=None),
            "end": datetime(2023, 10, 20, 20, 30, tzinfo=timezone.utc),
        },
        # CML filtering
        "cml": {
            "min_length": 0.5,
            "max_length": 100,
        },
        # db settings for historic calculation
        "historic": {
            "skip_influx": True,
            "skip_mariadb": True,
        },
        # user info for folder names and link selection (list of IDs)
        "user_info": {
            "folder_name": "kraken",
            "links_id": [i for i in range(1, 1000)],
        },
    }

    run_hist_calc(cfg)
