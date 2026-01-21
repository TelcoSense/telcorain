from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from influxdb_client import InfluxDBClient

from _case_study_links import LINKS_BRNO, LINKS_PRAGUE
from _case_study_run import main

URL_PUBLIC = "http://192.168.64.168:8087"
TOKEN_PUBLIC_READ = "Vt7QOW_3-sluvUvtBXVeX3ivr1L0DGoNhriQn_GogiSEXtYDo3qs6_htvb2EEs1eDGv-mpdym8iFTdX8dfy6_w=="
ORG = "vut"

client_public = InfluxDBClient(url=URL_PUBLIC, token=TOKEN_PUBLIC_READ, org=ORG)

######### BRNO case 1 ########### 10min
cfg_brno_1_10min = {
    # time setting
    "time": {
        "step": 10,
        "output_step": 10,
        "start": datetime(2025, 7, 6, 16, 0, tzinfo=None),
        "end": datetime(2025, 7, 6, 23, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": True},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/brno_1_10min",
        "links_id": LINKS_BRNO,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_brno_1_10min)

######### BRNO case 1 ########### 5min
cfg_brno_1_5min = {
    # time setting
    "time": {
        "step": 5,
        "output_step": 5,
        "start": datetime(2025, 7, 6, 16, 0, tzinfo=None),
        "end": datetime(2025, 7, 6, 23, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": True},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/brno_1_5min",
        "links_id": LINKS_BRNO,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_brno_1_5min)

######### BRNO case 2 ########### 10min
cfg_brno_2_10min = {
    # time setting
    "time": {
        "step": 10,
        "output_step": 10,
        "start": datetime(2025, 9, 10, 10, 0, tzinfo=None),
        "end": datetime(2025, 9, 11, 3, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": True},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/brno_2_10min",
        "links_id": LINKS_BRNO,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_brno_2_10min)

######### BRNO case 2 ########### 5min
cfg_brno_2_5min = {
    # time setting
    "time": {
        "step": 5,
        "output_step": 5,
        "start": datetime(2025, 9, 10, 10, 0, tzinfo=None),
        "end": datetime(2025, 9, 11, 3, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": True},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/brno_2_5min",
        "links_id": LINKS_BRNO,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_brno_2_5min)

######### PRAGUE case 1 ########### 10min
cfg_prague_1_10min = {
    # time setting
    "time": {
        "step": 10,
        "output_step": 10,
        "start": datetime(2025, 8, 29, 18, 0, tzinfo=None),
        "end": datetime(2025, 8, 30, 3, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": False},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/prague_1_10min",
        "links_id": LINKS_PRAGUE,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_prague_1_10min)

######### PRAGUE case 1 ########### 5min
cfg_prague_1_5min = {
    # time setting
    "time": {
        "step": 5,
        "output_step": 5,
        "start": datetime(2025, 8, 29, 18, 0, tzinfo=None),
        "end": datetime(2025, 8, 30, 3, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": False},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/prague_1_5min",
        "links_id": LINKS_PRAGUE,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_prague_1_5min)

######### PRAGUE case 2 ########### 10min
cfg_prague_2_10min = {
    # time setting
    "time": {
        "step": 10,
        "output_step": 10,
        "start": datetime(2025, 9, 5, 18, 0, tzinfo=None),
        "end": datetime(2025, 9, 6, 8, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": False},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/prague_2_10min",
        "links_id": LINKS_PRAGUE,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_prague_2_10min)

######### PRAGUE case 2 ########### 5min
cfg_prague_2_5min = {
    # time setting
    "time": {
        "step": 5,
        "output_step": 5,
        "start": datetime(2025, 9, 5, 18, 0, tzinfo=None),
        "end": datetime(2025, 9, 6, 8, 0, tzinfo=None),
    },
    "setting": {
        "dry_as_nan": False,
        "write_influx_intensities": False,
    },
    # CML filtering
    "cml": {"min_length": 0.5, "max_length": 100, "exclude_cmls": False},
    # user info for folder names and link selection (list of IDs)
    "user_info": {
        "folder_name": "./case_study_runs/prague_2_5min",
        "links_id": LINKS_PRAGUE,
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
        "idw_near": 12,
        "idw_dist_m": 20000.0,
    },
    "hour_sum": {
        "enabled": True,
        "write_influx_sum": False,
    },
    "rendering": {
        "is_crop_enabled": True,
    },
}
main(cfg_prague_2_5min)
