# TelcoRain

TelcoRain is a Python pipeline for estimating rainfall from commercial microwave links (CMLs).  
It reads metadata from MariaDB, timeseries data from InfluxDB, supports both real-time and historic processing, computes spatial interpolation using IDW, hourly rainfall sums, optional temperature-based compensation, and export results to PNG, JSON, and InfluxDB. The system is designed as a configurable processing pipeline driven by an INI configuration file. Individual processing stages (data loading, wet–dry detection, rainfall estimation, interpolation, visualization) can be enabled or disabled without modifying the code.
---

## Main features

- InfluxDB 2.x data source
- Real-time and historic processing modes
- Rainfall intensity estimation (mm/h)
- Hourly rainfall accumulation (mm)
- Spatial interpolation using IDW (Inverse Distance Weighting)
- Optional wet–dry detection:
  - threshold-based
  - rolling statistics
  - MLP / CNN
- Optional temperature filtering and compensation
- Output formats:
  - PNG maps
  - NPY raw grids
  - JSON metadata
  - InfluxDB time series
- Geographic masking using GeoJSON
- Mercator (EPSG:3857) and lon/lat grids
- Single pipeline for batch, realtime, and web usage

---

## Processing Pipeline Overview

Conceptual processing flow:

```
InfluxDB
  ↓
load_data_from_influxdb()
  ↓
convert_to_link_datasets()
  ↓
wet–dry detection (rolling window / CNN)
  ↓
rainfall intensity estimation (R)
  ↓
hour-sum rolling window (optional)
  ↓
generate_rainfields()   (IDW interpolation)
  ↓
Writer
  - PNG
  - JSON
  - InfluxDB
```

---

## Installation

### 1. Prerequisites
- Conda/Miniconda  
- Python 3.10  
- InfluxDB 2.x  
- MariaDB  

### 2. Create the environment
```bash
conda env create -f env_info/environment_linux.yml
conda activate telcorain_env
```

### 3. Configure `config.ini`
Sections:
- `[influx2]` → URL, token, buckets  
- `[mariadb]` → login, metadata DB  
- `[directories]`, `[realtime]`, `[cml]`, `[time]`, `[wet_dry]`, `[waa]`, `[interp]`, `[rendering]`, `[logging]`

### 4. Build of raincolor using cython
There is an optimized version of colorgrid matching that requires prebuild cython files. Especially on linux, you have to build essentials:

```bash
sudo apt-get install -y build-essential python3-dev
```

then cython:

```bash
conda install cython
```

and then run the setup:

```bash
python telcorain/cython/setup.py build_ext --inplace
```

## Execution Modes

### Real-time Mode

- Runs in a continuous loop
- Queries InfluxDB in a moving time window (e.g. 1 day)
- Processes only newly available time steps
- Writes incremental results to InfluxDB

### Historic Mode

- One-shot batch processing
- Fixed start and end time
- Supports warm-up for rolling windows and CNN-based models
- Writes full output time series
- Intended for reanalysis, validation, and reporting
---

## Core components

---

## InfluxDB Data Access

InfluxDB queries are executed using Flux with:

- server-side aggregateWindow
- server-side pivot to wide format
- chunked IP queries to avoid query size limits

Logical fields produced by the query:

- rx_power
- tx_power
- temperature (optional)

### Conditional Temperature Fetching

Temperature data is fetched only if required.

Temperature is queried if and only if at least one of the following is enabled:

- wet_dry.is_temp_filtered = true
- wet_dry.is_temp_compensated = true

If neither option is enabled, temperature is not queried at all, which reduces
InfluxDB load and speeds up processing.

---

## Wet–Dry Detection

Wet–dry detection is optional and configurable.

Supported approaches include:

- fixed thresholding
- rolling statistics
- neural networks (MLP / CNN)

All parameters are defined in the [wet_dry] configuration section.

---

## Rainfall Intensity Estimation

For each microwave link and time step, rainfall intensity R (mm/h) is estimated.

If a link has multiple channels, channel values are averaged.

---

## Hourly Rainfall Sums

The optional hour_sum feature computes accumulated rainfall using a rolling window
(e.g. 60 minutes).

Results are stored as R_hour_sum (mm).

---

## Spatial Interpolation (IDW)

Spatial interpolation from link values to a regular grid is performed using
Inverse Distance Weighting (IDW).

---

## Outputs

- PNG rainfall maps
- JSON metadata
- NPY raw grids (optional)
- InfluxDB time series
---

## Running the pipeline

### Realtime mode
```bash
python run_cli.py
```

Loop:
- cleanup  
- calculation  
- writer  
- sleep  

### Historic mode
```bash
python run_historic.py
```

Configure:
- time window  
- link IDs  
- interpolation parameters  
- WAA & wet/dry settings  

---

## Acknowledgements

This output was financed through the project Precipitation Detection and Quantification System Based on Networks of Microwave Links (SS06020416) is co-funded with state support from the Technology Agency of the Czech Republic under the Environment for Life Programme. The project was further funded within the National Recovery Plan from the European Recovery and Resilience Facility.

<p align="center">
  <img src="assets/tacr.png" alt="Technology Agency of the Czech Republic" height="64" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eu.png" alt="European Union" height="64" />
</p>