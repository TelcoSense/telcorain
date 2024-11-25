from telcorain.handlers.historic_writer import HistoricWriter
from telcorain.procedures.utils.helpers import utc_datetime

start = utc_datetime(year=2024, month=9, day=10)
end = utc_datetime(year=2024, month=9, day=20)

# time_diff = (end - start).total_seconds() * 1000

calculation_params = {
    "start": start,
    "end": end,
    "step": 5,
    # "time_diff": time_diff,
    "is_cnn_enabled": False,
    "is_external_filter_enabled": False,
    "external_filter_params": None,
    "rolling_hours": 6.0,
    "rolling_values": 120,
    "wet_dry_deviation": 0.8,
    "baseline_samples": 5,
    "interpol_res": 0.01,
    "idw_power": 1,
    "idw_near": 15,
    "idw_dist": 0.1,
    "output_step": 10,
    "min_rain_value": 0.1,
    "is_only_overall": False,
    "is_output_total": True,
    "is_pdf": False,
    "is_png": True,
    "is_dummy": False,
    "map_file": "brno.png",
    "animation_speed": 1000,
    "waa_schleiss_val": 2.3,
    "waa_schleiss_tau": 15.0,
    "is_temp_compensated": False,
    "correlation_threshold": 0.7,
    "realtime_timewindow": "Past 3 h",
    "is_realtime": False,
    "is_temp_filtered": False,
    "is_output_write": False,
    "is_history_write": False,
    "is_force_write": False,
    "is_influx_write_skipped": False,
    "is_window_centered": True,
    "retention": 336,
    "X_MIN": 16.5188567,
    "X_MAX": 16.7054181,
    "Y_MIN": 49.143475,
    "Y_MAX": 49.2305394,
}


historic_writer = HistoricWriter(
    calculation_params=calculation_params,
)

historic_writer.calculate_rain_grids()
