# TelcoRain

TelcoRain is a Python pipeline for estimating rainfall from commercial microwave link (CML) data.
It reads link metadata from MariaDB, reads telemetry from InfluxDB, classifies wet and dry periods,
estimates rainfall intensity on each link, interpolates the link values to spatial grids, and exports
the results to PNG, JSON, NPY, and InfluxDB outputs.

The project is designed around a configurable processing pipeline driven by `config.ini`. The same core
calculation code is used for realtime runs, historic backfills, and web-triggered jobs.

The output is used in the [TelcoSense platform](https://telcosense.cz/rain).

---

## What The Pipeline Does

- Loads CML metadata from MariaDB.
- Loads signal time series from InfluxDB 2.x.
- Builds per-link datasets from the raw IP-based telemetry.
- Performs optional wet/dry detection.
- Applies optional temperature filtering or compensation.
- Computes rainfall intensity `R` in mm/h.
- Computes optional rolling hour-sum rainfall in mm.
- Interpolates link values to a spatial grid using IDW.
- Writes map products and time series outputs.

---

## Main Features

- Realtime processing loop for operational rainfall maps.
- Historic processing mode for backfills and analysis.
- Optional web-triggered calculation mode.
- Threshold, rolling-statistics, and CNN/MLP wet-dry detection paths.
- Optional temperature-aware signal filtering/compensation.
- IDW interpolation in lon/lat or EPSG:3857 Mercator coordinates.
- Geographic masking using GeoJSON.
- PNG map rendering plus optional raw NPY grid export.
- JSON sidecar metadata for downstream consumers.
- InfluxDB export of link-level rainfall time series.

---

## Processing Flow

```text
MariaDB metadata
    |
    v
InfluxDB telemetry query
    |
    v
load_data_from_influxdb()
    |
    v
convert_to_link_datasets()
    |
    v
wet/dry detection
    |
    v
rainfall estimation
    |
    v
optional hour-sum computation
    |
    v
generate_rainfields()
    |
    v
Writer
```

At a high level, the important stages are:

- `telcorain/dataprocessing.py`: data loading and conversion from IP-level telemetry to CML datasets.
- `telcorain/procedures/wet_dry/`: wet/dry classification logic.
- `telcorain/procedures/rain/rain_calculation.py`: attenuation, WAA, and rain-rate computation.
- `telcorain/procedures/rain/rainfields_generation.py`: spatial interpolation and hour-sum grids.
- `telcorain/writer.py`: PNG, JSON, NPY, and InfluxDB output writing.

---

## Execution Modes

### Realtime mode

`run_cli.py` runs the operational loop.

Current realtime behavior:

- Keeps an in-memory rolling raw-data buffer for the configured time window.
- Fetches only an overlapping tail plus new samples instead of querying the full window every cycle.
- Periodically forces a full raw-data refresh to resynchronize late or backfilled data.
- Periodically reloads MariaDB metadata so added or removed links can be picked up without restarting.
- Reuses a persistent writer instance so expensive static assets such as the polygon mask are cached.
- Interpolates and writes only newly available timesteps instead of regenerating the full output window.

Typical loop:

1. Cleanup old outputs if enabled.
2. Refresh metadata when configured to do so.
3. Fetch incremental InfluxDB data.
4. Run rainfall calculation.
5. Generate new rainfields.
6. Write outputs for new timesteps.
7. Sleep until the next cycle.

### Historic mode

`run_historic.py` runs a one-shot calculation for a fixed time interval.

Historic mode is intended for:

- backfills
- validation
- reanalysis
- web-driven custom jobs

Historic runs use the full requested interval and can include warm-up samples for rolling windows and
CNN-based wet/dry workflows.

### Web mode

`run_web.py` executes a calculation from a JSON configuration payload and writes results to web-facing
output directories.

---

## Installation

### Prerequisites

- Conda or Miniconda
- Python 3.10
- InfluxDB 2.x
- MariaDB

### Create the environment

Linux:

```bash
conda env create -f env_info/environment_linux.yml
conda activate telcorain_env
```

Windows:

```bash
conda env create -f env_info/environment_win.yml
conda activate telcorain_env
```

### Configure `config.ini`

Start from:

```text
configs/config.ini.dist
```

Then create:

```text
configs/config.ini
```

The most important sections are:

- `[influx2]`: InfluxDB URL, token, organization, and buckets.
- `[mariadb]`: MariaDB connection and metadata database settings.
- `[time]`: base input step, output step, and historic start/end range.
- `[realtime]`: realtime window, retention window, and metadata refresh cadence.
- `[cml]`: link filtering such as minimum and maximum link length.
- `[wet_dry]`: wet/dry classification settings.
- `[temp]`: temperature filtering and compensation settings.
- `[waa]`: wet-antenna attenuation method.
- `[interp]`: interpolation grid and IDW settings.
- `[raingrids]`: rainfield thresholds and overall intensity scoring.
- `[directories]`: output folders and save/cleanup flags.
- `[rendering]`: GeoJSON mask and base map settings.
- `[logging]`: logging level.

---

## Optional Cython Build

The project includes an optimized color-mapping implementation in `telcorain/cython/`.

On Linux, install build tools first:

```bash
sudo apt-get install -y build-essential python3-dev
```

Then install Cython and build the extension:

```bash
conda install cython
python telcorain/cython/setup.py build_ext --inplace
```

---

## Running The Project

### Realtime CLI

```bash
python run_cli.py
```

Optional first run with the retention window:

```bash
python run_cli.py --first
```

### Historic run

```bash
python run_historic.py
```

### Web-triggered run

```bash
python run_web.py --cfg "{...json payload...}"
```

---

## InfluxDB Data Access

InfluxDB queries are executed with Flux and currently use:

- server-side `aggregateWindow`
- server-side pivot to a wide DataFrame
- chunked IP batches to avoid oversized Flux queries
- conditional temperature loading
- incremental buffering in realtime mode

Logical fields returned by the wide query:

- `_time`
- `agent_host`
- `rx_power`
- `tx_power`
- `temperature` when requested

### Conditional temperature fetching

Temperature is queried only when it is needed later in the pipeline.

It is fetched when at least one of these is enabled:

- `temp.is_temp_filtered = true`
- `temp.is_temp_compensated = true`

If both are disabled, temperature is omitted from the query to reduce load and memory use.

### Realtime buffering

Realtime mode keeps a rolling DataFrame cache of the current processing window.

The cache is:

- merged with newly fetched tail data
- deduplicated by `(_time, agent_host)`
- trimmed to the configured realtime window
- reset when the metadata selection changes

This design allows the number of active CMLs to change over time without assuming a fixed matrix shape.

---

## Wet/Dry Detection

Wet/dry detection is optional and configurable.

Supported paths include:

- simple thresholding
- rolling standard deviation logic
- CNN or MLP-based inference
- custom 30-second CNN preprocessing path for higher-resolution wet/dry classification

The relevant parameters live in `[wet_dry]`.

---

## Rainfall Estimation

For each link and time step, the pipeline:

1. preprocesses `tsl` and `rsl`
2. builds `trsl`
3. estimates a baseline
4. applies the configured WAA method
5. converts attenuation to rainfall intensity `R`

If a link has two channels, the downstream spatial interpolation uses the mean channel value.

Supported WAA methods are configured in `[waa]`.

---

## Hour-Sum Rainfall

If `[hour_sum].enabled = true`, the pipeline also computes rolling accumulated rainfall over the configured
window, typically 60 minutes.

This produces `R_hour_sum` in millimeters and can be:

- rendered to PNG
- written to JSON
- optionally exported to InfluxDB

---

## Spatial Interpolation

Spatial interpolation is performed with inverse distance weighting (IDW).

Key options:

- `interp.use_mercator`
- `interp.grid_nx`
- `interp.grid_ny`
- `interp.grid_step_m`
- `interp.idw_power`
- `interp.idw_near`
- `interp.idw_dist` or `interp.idw_dist_m`

Supported coordinate modes:

- lon/lat grid
- EPSG:3857 Mercator grid

If rendering crop is enabled, the output grid is masked using the configured GeoJSON polygon.

---

## Outputs

Depending on configuration, the pipeline can produce:

- PNG intensity maps
- PNG hour-sum maps
- JSON metadata for each frame
- raw NPY grids
- InfluxDB link-level time series
- log files

Main output folders are configured under `[directories]`.

---

## Important Realtime Notes

- Realtime map generation is append-only for newly detected timesteps.
- The internal raw-data buffer can resync late samples, but already written older PNG/JSON outputs are not
  automatically regenerated.
- Metadata changes are detected on a run cadence controlled by:

```ini
[realtime]
metadata_refresh_interval_runs=60
```

Set `metadata_refresh_interval_runs=0` to disable periodic metadata reloads.

---

## Repository Layout

```text
configs/                Configuration templates
assets/                 GeoJSON masks and base images
cml_info/               Link exclusion lists and related metadata files
env_info/               Conda environments and requirements
logs/                   Runtime logs
telcorain/
  calculation.py        Main orchestration logic
  dataprocessing.py     Influx loading and dataset conversion
  writer.py             Output writing
  database/             InfluxDB and MariaDB access
  procedures/
    wet_dry/            Wet/dry classification logic
    rain/               Rain-rate and rainfield generation logic
run_cli.py              Realtime CLI entry point
run_historic.py         Historic runner
run_web.py              Web-triggered runner
```

---

## Acknowledgements

This output was financed through the project Precipitation Detection and Quantification System Based on
Networks of Microwave Links (SS06020416), co-funded with state support from the Technology Agency of the
Czech Republic under the Environment for Life Programme. The project was further funded within the National
Recovery Plan from the European Recovery and Resilience Facility.

<p align="center">
  <img src="assets/tacr.png" alt="Technology Agency of the Czech Republic" height="64" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/eu.png" alt="European Union" height="64" />
</p>
