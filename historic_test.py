import warnings
from datetime import datetime, timedelta

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers.logging_handler import setup_init_logging, logger
from telcorain.handlers.writer import Writer
from telcorain.procedures.calculation import CalculationHistoric
from telcorain.procedures.utils.helpers import create_cp_dict, select_all_links

warnings.simplefilter(action="ignore", category=FutureWarning)
setup_init_logging(logger)

# load calculation params dict and global config dict
cp = create_cp_dict(path="./configs/config_calc.ini", format=True)
config = create_cp_dict(path="./configs/config.ini", format=False)
# create sql manager and filter out short links
sql_man = SqlManager(min_length=cp["cml"]["min_length"])
# load link definitions from MariaDB
links = sql_man.load_metadata()
# select all links
selected_links = select_all_links(links=links)
# define calculation object
calculation = CalculationHistoric(
    influx_man=influx_man,
    results_id=0,
    links=links,
    selection=selected_links,
    cp=cp,
)
# run the calculation
calculation.run()

start_time = datetime.now()
output_delta = timedelta(minutes=cp["time"]["output_step"])
since_time = start_time - output_delta

# create the writer object and write the results to disk
writer = Writer(
    sql_man=sql_man,
    influx_man=influx_man,
    write_historic=True,
    skip_influx=True,
    skip_sql=True,
    since_time=since_time,
    cp=cp,
    config=config,
)

writer.push_results(
    rain_grids=calculation.rain_grids,
    x_grid=calculation.x_grid,
    y_grid=calculation.y_grid,
    calc_dataset=calculation.calc_data_steps,
)
