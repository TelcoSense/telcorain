from telcorain.handlers.historic_writer import HistoricWriter
from telcorain.procedures.utils.helpers import utc_datetime

start = utc_datetime(year=2024, month=9, day=18, hour=18)
end = utc_datetime(year=2024, month=9, day=19)

# time_diff = (end - start).total_seconds() * 1000

cp = {
    "start": start,
    "end": end,
    "step": 10,
    # "time_diff": time_diff,
    "is_cnn_enabled": False,
    "is_external_filter_enabled": False,
    "external_filter_params": None,
    "rolling_hours": 1.0,
    "rolling_values": 30,
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
    "X_MIN": 12.0905,
    "X_MAX": 18.8591,
    "Y_MIN": 48.5525,
    "Y_MAX": 51.0557,
}


historic_writer = HistoricWriter(cp=cp)

historic_writer.write_raingrids()
