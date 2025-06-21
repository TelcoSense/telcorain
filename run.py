import warnings
from datetime import datetime, timedelta, timezone

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers.logging_handler import setup_init_logging, logger
from telcorain.handlers.writer import Writer
from telcorain.procedures.calculation import CalculationHistoric
from telcorain.procedures.utils.helpers import create_cp_dict, select_all_links

warnings.simplefilter(action="ignore", category=FutureWarning)
setup_init_logging(logger)


def run_hist_calc(cp: dict):
    # load global config dict
    config = create_cp_dict(path="./configs/config.ini", format=False)

    # add cp config info nonrelevant for web app
    cp.update(
        {
            "external_filter": {
                "url": "http://192.168.64.166/chmi/data/CZRAD_10m",
                "file_prefix": "FILTER_LAYER_",
                "radius": 20,
                "pixel_threshold": 15,
                "default_return": False,
                "max_history_lookups": 3,
                "img_x_min": 11.28,
                "img_x_max": 20.765,
                "img_y_min": 48.05,
                "img_y_max": 52.165,
            },
            "realtime": {
                "is_realtime": False,
                "realtime_timewindow": "7d",
                "realtime_optimization": False,
                "is_output_write": True,
                "is_history_write": False,
                "is_force_write": False,
                "is_influx_write_skipped": False,
                "is_sql_write_skipped": False,
            },
            "wet_dry": {
                "is_mlp_enabled": False,
                "rolling_hours": 1.0,
                "rolling_values": 10,
                "wet_dry_deviation": 0.8,
                "is_window_centered": False,
                "baseline_samples": 5,
            },
            "waa": {
                "waa_schleiss_val": 2.3,
                "waa_schleiss_tau": 15.0,
            },
            "temp": {
                "is_temp_filtered": False,
                "is_temp_compensated": False,
                "correlation_threshold": 0.7,
            },
            "interp": {
                "interp_res": 0.01,
                "idw_power": 1,
                "idw_near": 35,
                "idw_dist": 1.5,
            },
            "raingrids": {
                "min_rain_value": 0.1,
                "is_only_overall": False,
                "is_output_total": False,
                "is_external_filter_enabled": False,
            },
            "limits": {
                "x_min": 12.0905,
                "x_max": 18.8591,
                "y_min": 48.5525,
                "y_max": 51.0557,
            },
            "rendering": {
                "is_crop_enabled": True,
                "geojson_file": "czechia.json",
                "map": "plain_czechia.png",
            },
        },
    )

    start_time = datetime.now()
    logger.info("Starting the historic calculation at: %s", start_time)

    # create sql manager and filter out short links
    sql_man = SqlManager()
    # load link definitions from MariaDB
    links = sql_man.load_metadata(
        ids=cp["user_info"]["links_id"],
        min_length=cp["cml"]["min_length"],
        max_length=cp["cml"]["max_length"],
        exclude_ids=True,
    )

    # select all links
    selected_links = select_all_links(links=links)
    # define calculation object
    calculation = CalculationHistoric(
        influx_man=influx_man,
        results_id=0,
        links=links,
        selection=selected_links,
        cp=cp,
        compensate_historic=config["setting"]["compensate_historic"],
    )
    # run the calculation
    calculation.run()

    # create the writer object and write the results to disk
    writer = Writer(
        sql_man=sql_man,
        influx_man=influx_man,
        write_historic=True,
        skip_influx=cp["historic"]["skip_influx"],
        skip_sql=True,
        cp=cp,
        config=config,
    )

    writer.push_results(
        rain_grids=calculation.rain_grids,
        x_grid=calculation.x_grid,
        y_grid=calculation.y_grid,
        calc_dataset=calculation.calc_data_steps,
    )


if __name__ == "__main__":
    # this comes from the web app settings
    cp = {
        # time setting (probably dont change step and output_step)
        "time": {
            "step": 10,
            "output_step": 10,
            "start": datetime(2023, 10, 20, 3, 30, tzinfo=timezone.utc),
            "end": datetime(2023, 10, 20, 20, 30, tzinfo=timezone.utc),
        },
        # CML filtering
        "cml": {
            "min_length": 0.5,
            "max_length": 100,
        },
        # db settings for historic calculation
        "historic": {
            "skip_influx": False,
        },
        # user info for folder names and link selection (list of IDs)
        "user_info": {
            "folder_name": "kraken",
            "links_id": [i for i in range(1, 1000)],
        },
    }

    run_hist_calc(cp)
