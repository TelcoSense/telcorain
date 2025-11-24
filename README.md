# TelcoRain

TelcoRain is a Python pipeline for estimating rainfall from commercial microwave links (CMLs).  
It reads CML power/temperature data from InfluxDB, uses metadata from MariaDB, computes rain rates per link, interpolates them to spatial rainfields, and writes gridded outputs (`.npy`, `.png`) plus optional time-series back to InfluxDB.

---

## Main features

- InfluxDB 2.x reader with server-side aggregation and pivoting to a clean wide DataFrame.
- MariaDB metadata loader for link coordinates, frequencies, polarisation, etc.
- Per-link rain rate computation using:
  - Baseline estimation (pycomlink)
  - Wet/dry classification (CNN or rolling STD)
  - Wet-antenna attenuation (Schleiss, Leijnse, Pastorek)
  - k–R relation from pycomlink
- Spatial interpolation using inverse distance weighting (IDW)
- Realtime CLI (continuous loop) and historic “one-shot” computation
- Optional cropping of outputs using a GeoJSON mask
- Configurable via `configs/config.ini` + Python overrides

---

## Repository layout

```
telcorain/
├─ assets/
│  ├─ brno.png
│  ├─ czechia.json
│  └─ plain_czechia.png
├─ cml_info/
│  └─ invalid_cmls.csv
├─ configs/
│  └─ config.ini
├─ env_info/
│  ├─ environment_base.yml
│  ├─ environment_full.yml
│  └─ requirements_full.txt
├─ logs
├─ telcorain/
│  ├─ cython
│  ├─ database/
│  │  ├─ influx_manager.py
│  │  └─ sql_manager.py
│  ├─ procedures/
│  │  ├─ rain/
│  │  │  ├─ rain_calculation.py
│  │  │  ├─ rainfields_generation.py
│  │  │  └─ temperature_compensation.py
│  │  ├─ wet_dry/
│  │  └─ exceptions.py
│  ├─ __init__.py
│  ├─ calculation.py
│  ├─ dataprocessing.py
│  ├─ handlers.py
│  ├─ helpers.py
│  ├─ writer.py
├─ .gitignore 
├─ README..md
├─ run_cli.py
├─ run_historic.py
└─ README.md
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
conda env create -f env_info/environment_full.yml
conda activate telcorain_env
```

### 3. Configure `config.ini`
Sections:
- `[influx2]` → URL, token, buckets  
- `[mariadb]` → login, metadata DB  
- `[directories]`, `[realtime]`, `[cml]`, `[time]`, `[wet_dry]`, `[waa]`, `[interp]`, `[rendering]`, `[logging]`

---

## Core components

---

## 1. InfluxManager (`database/influx_manager.py`)

Responsible for reading CML time series from InfluxDB and optionally writing rain rates.

### Key methods

- **query_units(...)**
  - Executes window queries  
  - Returns a pivoted wide DataFrame  

- **query_units_realtime(...)**
  - Converts window strings ("1h", "1d", "7d")  
  - Calls `query_units`

- **write_points(points, bucket)**
  - Writes rain-rate time series to an Influx bucket

---

## 2. SqlManager (`database/sql_manager.py`)

Loads metadata from MariaDB.

### Key methods
- **load_metadata(...)**
  - Loads link info, frequencies, polarization, etc.  
  - Applies filters by ID, length, exclude-list  

- **get_last_raingrid() / insert_raingrid()**
  - Backwards compatibility (rarely needed)

---

## 3. Data loading & conversion (`dataprocessing.py`)

### Functions

- **load_data_from_influxdb(...)**
  - Builds IP list  
  - Fetches data  
  - Returns `(df, missing_links, ips)`

- **convert_to_link_datasets(...)**
  - Groups by agent host  
  - Builds xarray Datasets per MW link  
  - Creates two channels per link

---

## 4. Rain rate computation (`procedures/rain/rain_calculation.py`)

### Pipeline steps
- Clean & interpolate powers  
- Remove temperature‑correlated links  
- Wet/dry:
  - CNN-based  
  - or rolling STD  
- Baseline estimation  
- Wet‑antenna attenuation  
- Compute final rain intensity  
- Produce xarray Dataset

---

## 5. Rainfields generation (`procedures/rain/rainfields_generation.py`)

### Steps
- Build grid from limits & resolution  
- Apply IDW interpolation  
- Generate accumulated rainfall  
- Create animation frames  
- Crop if enabled  

### Returns
- **Realtime** → `(rain_grids, calc_data_steps, x_grid, y_grid, realtime_runs, last_time)`  
- **Historic** → `(rain_grids, calc_data_steps, x_grid, y_grid)`

---

## 6. Writer (`writer.py`)

Writes outputs to disk and optionally to InfluxDB.

### Methods
- **_write_raingrids**  
- **_write_timeseries_realtime**  
- **_write_timeseries_historic**  
- **push_results(...)**

---

## 7. Calculation (`calculation.py`)

Central orchestrator for one pipeline run.

### Workflow (`run(...)`)
1. Load data  
2. Compute rain rates  
3. Generate rainfields  
4. Update internal state  
5. Cleanup  
6. Logging with `@measure_time`

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

