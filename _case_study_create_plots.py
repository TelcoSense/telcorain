from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from influxdb_client import InfluxDBClient

from _case_study_links import BRNO_STATIONS, PRAGUE_STATIONS

# public influx for weather station data
URL_PUBLIC = "http://192.168.64.168:8087"
TOKEN_PUBLIC_READ = "Vt7QOW_3-sluvUvtBXVeX3ivr1L0DGoNhriQn_GogiSEXtYDo3qs6_htvb2EEs1eDGv-mpdym8iFTdX8dfy6_w=="
ORG = "vut"
client_public = InfluxDBClient(url=URL_PUBLIC, token=TOKEN_PUBLIC_READ, org=ORG)


BRNO_1_FOLDER = "./case_study_runs/brno_1_10min"
calc_dataset_brno_1 = xr.open_dataset(f"{BRNO_1_FOLDER}/outputs_web/calc_dataset.nc")

BRNO_2_FOLDER = "./case_study_runs/brno_2_10min"
calc_dataset_brno_2 = xr.open_dataset(f"{BRNO_2_FOLDER}/outputs_web/calc_dataset.nc")

PRAGUE_1_FOLDER = "./case_study_runs/prague_1_10min"
calc_dataset_prague_1 = xr.open_dataset(
    f"{PRAGUE_1_FOLDER}/outputs_web/calc_dataset.nc"
)

PRAGUE_2_FOLDER = "./case_study_runs/prague_2_10min"
calc_dataset_prague_2 = xr.open_dataset(
    f"{PRAGUE_2_FOLDER}/outputs_web/calc_dataset.nc"
)

print("Num CML BRNO 1:", calc_dataset_brno_1.sizes["cml_id"])
print("Num CML PRAGUE 1:", calc_dataset_prague_1.sizes["cml_id"])
print("Num CML PRAGUE 2:", calc_dataset_prague_2.sizes["cml_id"])


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def nearest_links_by_center(
    ds: xr.Dataset,
    station: dict,
    k: int = 3,
    *,
    min_valid: int = 3,
    require_rain: bool = True,
    rain_eps: float = 1e-6,  # consider > rain_eps as "rain present"
):
    """
    Return k nearest links (by center distance) that have usable R data
    over the dataset time range.

    Usable means:
      - at least `min_valid` finite samples after averaging over channel_id
      - and (optionally) at least one sample > `rain_eps` if `require_rain=True`
    """
    t0 = ds["time"].values[0]
    t1 = ds["time"].values[-1]
    Rw = ds["R"].sel(time=slice(t0, t1))

    # --- geometry / centers ---
    a_lat = ds["site_a_latitude"].values.astype(float)
    a_lon = ds["site_a_longitude"].values.astype(float)
    b_lat = ds["site_b_latitude"].values.astype(float)
    b_lon = ds["site_b_longitude"].values.astype(float)

    c_lat = 0.5 * (a_lat + b_lat)
    c_lon = 0.5 * (a_lon + b_lon)

    st_lat = float(station["lat"])
    st_lon = float(station["lon"])
    d = haversine_m(st_lat, st_lon, c_lat, c_lon)

    cml_ids = ds["cml_id"].values.astype(int)

    candidates = []
    for i, cml_id in enumerate(cml_ids):
        if not np.isfinite(d[i]):
            continue

        # average 2 directions -> one curve
        arr = (
            Rw.sel(cml_id=int(cml_id))
            .mean("channel_id", skipna=True)
            .values.astype(float)
        )

        finite = np.isfinite(arr)
        if int(finite.sum()) < int(min_valid):
            continue

        if require_rain:
            arr_f = arr[finite]
            if not np.any(arr_f > rain_eps):
                continue

        candidates.append((int(cml_id), float(d[i])))

    candidates.sort(key=lambda x: x[1])
    return candidates[:k]


def to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def unpack_series(series):
    times = [datetime.fromisoformat(p["time"]) for p in series]
    values = [p["value"] for p in series]
    return times, values


def to_intensity_mmph(series, window_minutes=10):
    factor = 60.0 / window_minutes
    times = [datetime.fromisoformat(p["time"]) for p in series]
    values = [(p["value"] or 0.0) * factor for p in series]
    return times, values


def station_intensity_series(result_sra10m, window_minutes=10) -> pd.Series:
    # returns pandas Series indexed by datetime, values mm/h
    times = [pd.to_datetime(p["time"]) for p in result_sra10m]
    vals = [(p["value"] or 0.0) * (60.0 / window_minutes) for p in result_sra10m]
    s = pd.Series(vals, index=pd.DatetimeIndex(times)).sort_index()
    return s


def link_intensity_series(ds: xr.Dataset, cml_id: int) -> pd.Series:
    # average 2 directions -> one curve
    da = ds["R"].sel(cml_id=cml_id).mean("channel_id", skipna=True)
    # xarray -> pandas Series with DatetimeIndex
    return da.to_series().sort_index()


def align_nearest(x: pd.Series, y: pd.Series, *, tol="1min") -> pd.DataFrame:
    """
    Align y to x by nearest timestamp within tolerance.
    Returns df with columns ['x','y'].
    Handles tz-aware vs tz-naive by normalizing both to UTC tz-aware.
    """
    x = x.dropna().sort_index()
    y = y.dropna().sort_index()

    # --- normalize datetime index types: make both UTC tz-aware ---
    ix = pd.to_datetime(x.index)
    iy = pd.to_datetime(y.index)

    if ix.tz is None:
        ix = ix.tz_localize("UTC")
    else:
        ix = ix.tz_convert("UTC")

    if iy.tz is None:
        iy = iy.tz_localize("UTC")
    else:
        iy = iy.tz_convert("UTC")

    x = pd.Series(x.to_numpy(), index=ix, name="x")
    y = pd.Series(y.to_numpy(), index=iy, name="y")

    df_x = x.to_frame()
    df_y = y.to_frame()

    df = pd.merge_asof(
        df_x,
        df_y,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tol),
    ).dropna()

    return df


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def query_ws_data(start, stop, gh_id):
    query = f"""
        from(bucket: "chmi_data")
        |> range(start: {start}, stop: {stop})
        |> filter(fn: (r) =>
        (r["_measurement"] == "T" or r["_measurement"] == "SRA10M") and r["_field"] == "{gh_id}")
        """.strip()
    result = {"T": [], "SRA10M": []}
    tables = client_public.query_api().query(query)
    for table in tables:
        for record in table.records:
            point = {
                "time": record.get_time().isoformat(),
                "value": record.get_value(),
            }
            measurement = record.get_measurement()
            if measurement in result:
                result[measurement].append(point)
    return result


def plot_station_vs_links_corr(
    *,
    calc_dataset: xr.Dataset,
    stations,
    start_dt: datetime,
    stop_dt: datetime,
    k_links: int = 4,
    tol: str = "6min",
    candidate_pool: int = 50,
    min_corr: float = 0.7,
    # skipping links
    skip_links_by_station=None,
    skip_links=None,
    # output
    out_dir: str = "./latex_figs",
    filename_prefix: str = "station_corr",
    linewidth_pt: float = 483.69684,
    # sizing (relative to width)
    ts_height_rel: float = 0.2,
    corr_height_rel: float = 0.8,
    # layout tweaks
    hgap: float = 0.08,
    xlabel_band: float = 0.060,
    clearance_pad: float = 0.010,
    # styling
    base_fs: float = 8.0,
    show: bool = False,
):
    """
    Same layout/visual style as plot_station_vs_links_corr_rolling1h_pdf, but for
    *intensity* plots (station_intensity_series + link_intensity_series).

    IMPORTANT: link selection (chosen links) is done using the SAME method as the
    rolling1h sum version (rolling 1h sums + correlation + min_corr + distance sort),
    so the selected link IDs match.
    """
    import os

    import matplotlib.dates as mdates

    os.makedirs(out_dir, exist_ok=True)

    # TeX pt -> inch for matplotlib
    fig_width_in = float(linewidth_pt) / 72.27
    ts_height_in = ts_height_rel * fig_width_in
    corr_height_in = corr_height_rel * fig_width_in
    fig_height_in = ts_height_in + corr_height_in

    start = to_rfc3339(start_dt)
    stop = to_rfc3339(stop_dt)

    # normalize skip inputs
    global_skip = set(int(x) for x in (skip_links or []))

    # ---------- selection helpers (MATCH rolling1h version) ----------
    def station_hour_sum_series(result_sra10m) -> pd.Series:
        times = [pd.to_datetime(p["time"]) for p in result_sra10m]
        vals = [(p["value"] or 0.0) for p in result_sra10m]
        s = pd.Series(vals, index=pd.DatetimeIndex(times)).sort_index()
        return s.rolling(window=6, min_periods=6).sum()

    def link_hour_sum_series(ds: xr.Dataset, cml_id: int) -> pd.Series:
        if "R_hour_sum" in ds:
            da = ds["R_hour_sum"].sel(cml_id=cml_id)
            return da.to_series().sort_index()
        da = ds["R"].sel(cml_id=cml_id).mean("channel_id", skipna=True)
        s = da.to_series().sort_index()
        s_10m = s * (10.0 / 60.0)
        return s_10m.rolling(window=6, min_periods=6).sum()

    # ---------- plotting labels helpers (same as pretty function) ----------
    def link_freq_ghz(ds: xr.Dataset, cml_id: int) -> float:
        if "frequency" not in ds:
            return np.nan
        try:
            v = np.asarray(ds["frequency"].sel(cml_id=cml_id).values, float).ravel()
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else np.nan
        except Exception:
            return np.nan

    def link_len_km(ds: xr.Dataset, cml_id: int) -> float:
        if "length" in ds:
            try:
                L = float(ds["length"].sel(cml_id=cml_id).values)
                return L if np.isfinite(L) else np.nan
            except Exception:
                pass
        a_lat = float(ds["site_a_latitude"].sel(cml_id=cml_id).values)
        a_lon = float(ds["site_a_longitude"].sel(cml_id=cml_id).values)
        b_lat = float(ds["site_b_latitude"].sel(cml_id=cml_id).values)
        b_lon = float(ds["site_b_longitude"].sel(cml_id=cml_id).values)
        return float(haversine_m(a_lat, a_lon, b_lat, b_lon)) / 1000.0

    def ts_legend_label(ds: xr.Dataset, cml_id: int, dist_m: float) -> str:
        d_km = float(dist_m) / 1000.0
        L = link_len_km(ds, cml_id)
        f = link_freq_ghz(ds, cml_id)

        ds_ = rf"$d={d_km:.1f}\,\mathrm{{km}}$"
        Ls = rf"$L={L:.1f}\,\mathrm{{km}}$" if np.isfinite(L) else r"$L=?$"
        fs = rf"$f={f:.1f}\,\mathrm{{GHz}}$" if np.isfinite(f) else r"$f=?$"

        return f"KMS {cml_id}, {ds_}, {Ls}, {fs}"

    def corr_title_str(cml_id: int, r: float) -> str:
        return rf"KMS {cml_id}" + "\n" + rf"$r={r:.2f}$"

    figs = []
    saved = []

    for station in stations:
        # ----------- data -----------

        result = query_ws_data(start, stop, station["gh_id"])

        # PLOT series (intensity) — keep as-is
        s_station_int = station_intensity_series(
            result["SRA10M"], window_minutes=10
        ).dropna()

        # SELECTION series (rolling 1h sums) — to match rolling version
        s_station_sel = station_hour_sum_series(result["SRA10M"]).dropna()

        cands = nearest_links_by_center(
            calc_dataset,
            station,
            k=candidate_pool,
            require_rain=True,
        )

        # ----------- per-station skips -----------
        station_skip = set()
        if skip_links_by_station:
            try:
                station_skip |= set(
                    int(x)
                    for x in (skip_links_by_station.get(station["gh_id"], []) or [])
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station["gh_id"], []) or []
                )

            try:
                station_skip |= set(
                    int(x)
                    for x in (skip_links_by_station.get(station.get("name"), []) or [])
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station.get("name"), []) or []
                )

        skip = global_skip | station_skip
        if skip:
            cands = [
                (cml_id, dist_m)
                for (cml_id, dist_m) in cands
                if int(cml_id) not in skip
            ]

        # ----------- SELECT LINKS (MATCH rolling1h selection) -----------
        evaluated = []
        for cml_id, dist_m in cands:
            cml_id = int(cml_id)
            if cml_id in skip:
                continue

            s_link_sel = link_hour_sum_series(calc_dataset, cml_id).dropna()
            df_sel = align_nearest(s_station_sel, s_link_sel, tol=tol)
            r = pearson_r(df_sel["x"].to_numpy(), df_sel["y"].to_numpy())
            if not np.isfinite(r):
                continue
            if min_corr is not None and r < float(min_corr):
                continue

            # store r, but keep id/dist; df for plotting is computed later (intensity)
            evaluated.append((cml_id, float(dist_m), float(r)))

        evaluated.sort(key=lambda x: x[1])  # by distance (same as rolling)
        chosen_meta = evaluated[:k_links]

        if len(chosen_meta) < k_links:
            fallback = []
            chosen_ids = {c[0] for c in chosen_meta}
            for cml_id, dist_m in cands:
                cml_id = int(cml_id)
                if cml_id in skip or cml_id in chosen_ids:
                    continue

                s_link_sel = link_hour_sum_series(calc_dataset, cml_id).dropna()
                df_sel = align_nearest(s_station_sel, s_link_sel, tol=tol)
                r = pearson_r(df_sel["x"].to_numpy(), df_sel["y"].to_numpy())
                if not np.isfinite(r):
                    continue
                fallback.append((cml_id, float(dist_m), float(r)))

            # same fallback sort as rolling
            fallback.sort(key=lambda x: (-x[2], x[1]))
            chosen_meta.extend(fallback[: (k_links - len(chosen_meta))])

        # Now build "chosen" tuples WITH intensity-aligned df for plotting
        chosen = []
        for cml_id, dist_m, r in chosen_meta:
            s_link_int = link_intensity_series(calc_dataset, int(cml_id)).dropna()
            df_int = align_nearest(s_station_int, s_link_int, tol=tol)
            chosen.append((int(cml_id), float(dist_m), float(r), df_int))

        # ----------- figure + base geometry (same as rolling1h_pdf) -----------
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

        # normalized margins
        L = 0.07
        R = 0.995
        T = 0.95
        B = 0.08

        # TS axes: small, pinned to top
        total_rel = float(ts_height_rel) + float(corr_height_rel)
        usable_h = (T - B) - xlabel_band
        ts_h = max(0.06, min(0.30, usable_h * (float(ts_height_rel) / total_rel)))

        ts_y0 = T - ts_h
        ax_ts = fig.add_axes([L, ts_y0, (R - L), ts_h])

        # ----------- top TS plot (INTENSITY) -----------
        ax_ts.plot(
            s_station_int.index, s_station_int.values, label="Stanice", linewidth=1
        )

        for cml_id, dist_m, r, df in chosen:
            s_link_int = link_intensity_series(calc_dataset, int(cml_id)).dropna()
            ax_ts.plot(
                s_link_int.index,
                s_link_int.values,
                linestyle="--",
                alpha=0.85,
                linewidth=1,
                label=ts_legend_label(calc_dataset, int(cml_id), float(dist_m)),
            )

        ax_ts.set_title(station["name"], fontsize=base_fs + 1)
        ax_ts.set_ylabel("Intenzita [mm/h]", fontsize=base_fs)
        ax_ts.grid(True, alpha=0.3)

        ax_ts.tick_params(axis="both", labelsize=base_fs)
        ax_ts.tick_params(axis="x", pad=0.5)

        ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_ts.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
        ax_ts.set_xlabel("Čas [HH:MM]", fontsize=base_fs)

        ax_ts.legend(loc="upper right", fontsize=base_fs - 2, framealpha=0.9)

        # ----------- measure TS tight bbox -----------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        ts_tight = ax_ts.get_tightbbox(renderer)
        ts_tight_fig = ts_tight.transformed(fig.transFigure.inverted())
        ts_bottom = float(ts_tight_fig.y0)

        # ----------- correlation row geometry -----------
        avail_w = (R - L) - (k_links - 1) * hgap
        axw = avail_w / k_links
        axh = axw

        corr_y1 = ts_bottom - float(clearance_pad)
        corr_y0 = corr_y1 - axh

        min_corr_y0 = B + xlabel_band
        if corr_y0 < min_corr_y0:
            corr_y0 = min_corr_y0
            axh = corr_y1 - corr_y0
            axh = max(0.05, axh)
            axw = axh

            row_w = k_links * axw + (k_links - 1) * hgap
            if row_w > (R - L):
                hgap2 = max(0.01, hgap * 0.7)
                row_w = k_links * axw + (k_links - 1) * hgap2
                if row_w > (R - L):
                    axw = ((R - L) - (k_links - 1) * hgap2) / k_links
                    axh = axw
                hgap_use = hgap2
            else:
                hgap_use = hgap
        else:
            hgap_use = hgap

        # ----------- corr axes (INTENSITY scatter, r from SUM selection) -----------
        for j in range(k_links):
            x0 = L + j * (axw + hgap_use)
            ax_c = fig.add_axes([x0, corr_y0, axw, axh])

            if j >= len(chosen):
                ax_c.set_axis_off()
                continue

            cml_id, dist_m, r, df = chosen[j]
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()

            ax_c.scatter(x, y, s=10, alpha=0.65)

            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                m = float(np.nanmax(np.abs(np.concatenate([x[finite], y[finite]]))))
            else:
                m = 1.0
            if not np.isfinite(m) or m <= 0:
                m = 1.0

            ax_c.set_xlim(0.0, m)
            ax_c.set_ylim(0.0, m)
            ax_c.plot([0.0, m], [0.0, m], linewidth=1.0, alpha=0.9)

            ax_c.text(
                0.02,
                0.98,
                corr_title_str(int(cml_id), float(r)),
                transform=ax_c.transAxes,
                ha="left",
                va="top",
                fontsize=base_fs,
                linespacing=1.0,
            )

            ax_c.grid(True, alpha=0.3)
            ax_c.tick_params(axis="both", labelsize=base_fs, direction="out", pad=1.5)
            ax_c.set_xlabel("")
            ax_c.set_ylabel("")

        # ----------- shared labels (corr row) -----------
        left = L
        right = L + k_links * axw + (k_links - 1) * hgap_use
        xmid = 0.5 * (left + right)
        ymid = corr_y0 + 0.5 * axh

        sx = fig.supxlabel("Srážková intenzita stanice [mm/h]", fontsize=base_fs)
        sx.set_position((xmid, corr_y0 - 0.05))

        sy = fig.supylabel(
            "Srážková intenzita KMS [mm/h]", fontsize=base_fs, rotation=90
        )
        sy.set_position((left - 0.065, ymid))
        sy.set_in_layout(False)
        st_name = station.get("name", "station")
        out_path = f"{out_dir}/{filename_prefix}_{st_name}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.1)

        saved.append(out_path)
        figs.append(fig)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs, saved


def plot_station_vs_links_corr_rolling1h_pdf(
    *,
    calc_dataset: xr.Dataset,
    stations,
    start_dt: datetime,
    stop_dt: datetime,
    k_links: int = 4,
    tol: str = "6min",
    candidate_pool: int = 50,
    min_corr: float = 0.7,
    # NEW: skipping links
    skip_links_by_station=None,  # dict: { station["gh_id"] or station["name"] : {cml_id, ...}, ... }
    skip_links=None,  # global set/list of cml_id to skip for all stations
    # output
    out_dir: str = "./latex_figs",
    filename_prefix: str = "station_corr_rolling1h",
    linewidth_pt: float = 483.69684,  # LaTeX \linewidth in pt
    # sizing (relative to width)
    ts_height_rel: float = 0.2,
    corr_height_rel: float = 0.8,
    # layout tweaks
    hgap: float = 0.08,
    xlabel_band: float = 0.060,
    clearance_pad: float = 0.010,
    # styling
    base_fs: float = 8.0,
    show: bool = False,
):
    """
    Layout strategy:
    - Create TS axes near the top with desired small height.
    - Draw, measure its tight bbox (includes x tick labels).
    - Place corr row directly beneath that bbox with minimal clearance.
    - Corr axes are true squares by geometry (no set_aspect needed).

    Skipping strategy:
    - skip_links_by_station: map keyed by station["gh_id"] (preferred) or station["name"]
      with values being an iterable/set of CML IDs to exclude for that station.
    - skip_links: global iterable/set of CML IDs to exclude for all stations.
    """
    import os

    import matplotlib.dates as mdates

    os.makedirs(out_dir, exist_ok=True)

    # TeX pt -> inch for matplotlib
    fig_width_in = float(linewidth_pt) / 72.27
    ts_height_in = ts_height_rel * fig_width_in
    corr_height_in = corr_height_rel * fig_width_in
    fig_height_in = ts_height_in + corr_height_in

    start = to_rfc3339(start_dt)
    stop = to_rfc3339(stop_dt)

    # normalize skip inputs
    global_skip = set(int(x) for x in (skip_links or []))

    # ---------- series helpers ----------
    def station_hour_sum_series(result_sra10m) -> pd.Series:
        times = [pd.to_datetime(p["time"]) for p in result_sra10m]
        vals = [(p["value"] or 0.0) for p in result_sra10m]
        s = pd.Series(vals, index=pd.DatetimeIndex(times)).sort_index()
        return s.rolling(window=6, min_periods=6).sum()

    def link_hour_sum_series(ds: xr.Dataset, cml_id: int) -> pd.Series:
        if "R_hour_sum" in ds:
            da = ds["R_hour_sum"].sel(cml_id=cml_id)
            return da.to_series().sort_index()
        da = ds["R"].sel(cml_id=cml_id).mean("channel_id", skipna=True)
        s = da.to_series().sort_index()
        s_10m = s * (10.0 / 60.0)
        return s_10m.rolling(window=6, min_periods=6).sum()

    def link_freq_ghz(ds: xr.Dataset, cml_id: int) -> float:
        if "frequency" not in ds:
            return np.nan
        try:
            v = np.asarray(ds["frequency"].sel(cml_id=cml_id).values, float).ravel()
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else np.nan
        except Exception:
            return np.nan

    def link_len_km(ds: xr.Dataset, cml_id: int) -> float:
        if "length" in ds:
            try:
                L = float(ds["length"].sel(cml_id=cml_id).values)
                return L if np.isfinite(L) else np.nan
            except Exception:
                pass
        a_lat = float(ds["site_a_latitude"].sel(cml_id=cml_id).values)
        a_lon = float(ds["site_a_longitude"].sel(cml_id=cml_id).values)
        b_lat = float(ds["site_b_latitude"].sel(cml_id=cml_id).values)
        b_lon = float(ds["site_b_longitude"].sel(cml_id=cml_id).values)
        return float(haversine_m(a_lat, a_lon, b_lat, b_lon)) / 1000.0

    def ts_legend_label(ds: xr.Dataset, cml_id: int, dist_m: float) -> str:
        d_km = float(dist_m) / 1000.0
        L = link_len_km(ds, cml_id)
        f = link_freq_ghz(ds, cml_id)

        ds_ = rf"$d={d_km:.1f}\,\mathrm{{km}}$"
        Ls = rf"$L={L:.1f}\,\mathrm{{km}}$" if np.isfinite(L) else r"$L=?$"
        fs = rf"$f={f:.1f}\,\mathrm{{GHz}}$" if np.isfinite(f) else r"$f=?$"

        return f"KMS {cml_id}, {ds_}, {Ls}, {fs}"

    def corr_title_str(cml_id: int, r: float) -> str:
        return rf"KMS {cml_id}" + "\n" + rf"$r={r:.2f}$"

    figs = []
    saved = []

    for station in stations:
        # ----------- data -----------
        result = query_ws_data(start, stop, station["gh_id"])
        s_station = station_hour_sum_series(result["SRA10M"]).dropna()

        cands = nearest_links_by_center(
            calc_dataset,
            station,
            k=candidate_pool,
            require_rain=True,
        )

        # ----------- NEW: per-station skips -----------
        station_skip = set()
        if skip_links_by_station:
            # key by gh_id (preferred)
            try:
                station_skip |= set(
                    int(x)
                    for x in skip_links_by_station.get(station["gh_id"], []) or []
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station["gh_id"], []) or []
                )
            # also allow key by station name (optional convenience)
            try:
                station_skip |= set(
                    int(x)
                    for x in skip_links_by_station.get(station.get("name"), []) or []
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station.get("name"), []) or []
                )

        skip = global_skip | station_skip
        if skip:
            cands = [
                (cml_id, dist_m)
                for (cml_id, dist_m) in cands
                if int(cml_id) not in skip
            ]

        evaluated = []
        for cml_id, dist_m in cands:
            cml_id = int(cml_id)
            if cml_id in skip:
                continue
            s_link = link_hour_sum_series(calc_dataset, cml_id).dropna()
            df = align_nearest(s_station, s_link, tol=tol)
            r = pearson_r(df["x"].to_numpy(), df["y"].to_numpy())
            if not np.isfinite(r):
                continue
            if min_corr is not None and r < float(min_corr):
                continue
            evaluated.append((cml_id, float(dist_m), float(r), df))

        evaluated.sort(key=lambda x: x[1])  # by distance
        chosen = evaluated[:k_links]

        if len(chosen) < k_links:
            fallback = []
            chosen_ids = {c[0] for c in chosen}
            for cml_id, dist_m in cands:
                cml_id = int(cml_id)
                if cml_id in skip or cml_id in chosen_ids:
                    continue
                s_link = link_hour_sum_series(calc_dataset, cml_id).dropna()
                df = align_nearest(s_station, s_link, tol=tol)
                r = pearson_r(df["x"].to_numpy(), df["y"].to_numpy())
                if not np.isfinite(r):
                    continue
                fallback.append((cml_id, float(dist_m), float(r), df))
            fallback.sort(key=lambda x: (-x[2], x[1]))
            chosen.extend(fallback[: (k_links - len(chosen))])

        # ----------- figure + base geometry -----------
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

        # normalized margins
        L = 0.07
        R = 0.995
        T = 0.95
        B = 0.08

        # TS axes: small, pinned to top
        total_rel = float(ts_height_rel) + float(corr_height_rel)
        usable_h = (T - B) - xlabel_band
        ts_h = max(0.06, min(0.30, usable_h * (float(ts_height_rel) / total_rel)))

        ts_y0 = T - ts_h
        ax_ts = fig.add_axes([L, ts_y0, (R - L), ts_h])

        # ----------- top TS plot -----------
        ax_ts.plot(s_station.index, s_station.values, label="Stanice", linewidth=1)

        for cml_id, dist_m, r, df in chosen:
            s_link = link_hour_sum_series(calc_dataset, int(cml_id)).dropna()
            ax_ts.plot(
                s_link.index,
                s_link.values,
                linestyle="--",
                alpha=0.85,
                linewidth=1,
                label=ts_legend_label(calc_dataset, int(cml_id), float(dist_m)),
            )

        ax_ts.set_title(station["name"], fontsize=base_fs + 1)
        ax_ts.set_ylabel("1h úhrn [mm]", fontsize=base_fs)
        ax_ts.grid(True, alpha=0.3)

        ax_ts.tick_params(axis="both", labelsize=base_fs)
        ax_ts.tick_params(axis="x", pad=0.5)

        ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_ts.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
        ax_ts.set_xlabel("Čas [HH:MM]", fontsize=base_fs)

        ax_ts.legend(loc="upper right", fontsize=base_fs - 2, framealpha=0.9)

        # ----------- measure TS tight bbox (includes x tick labels) -----------
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        ts_tight = ax_ts.get_tightbbox(renderer)
        ts_tight_fig = ts_tight.transformed(fig.transFigure.inverted())
        ts_bottom = float(ts_tight_fig.y0)

        # ----------- correlation row geometry (square axes, no wasted space) -----------
        avail_w = (R - L) - (k_links - 1) * hgap
        axw = avail_w / k_links
        axh = axw

        corr_y1 = ts_bottom - float(clearance_pad)
        corr_y0 = corr_y1 - axh

        min_corr_y0 = B + xlabel_band
        if corr_y0 < min_corr_y0:
            corr_y0 = min_corr_y0
            axh = corr_y1 - corr_y0
            axh = max(0.05, axh)
            axw = axh

            row_w = k_links * axw + (k_links - 1) * hgap
            if row_w > (R - L):
                hgap2 = max(0.01, hgap * 0.7)
                row_w = k_links * axw + (k_links - 1) * hgap2
                if row_w > (R - L):
                    axw = ((R - L) - (k_links - 1) * hgap2) / k_links
                    axh = axw
                hgap_use = hgap2
            else:
                hgap_use = hgap
        else:
            hgap_use = hgap

        # ----------- corr axes -----------
        corr_axes = []
        for j in range(k_links):
            x0 = L + j * (axw + hgap_use)
            ax_c = fig.add_axes([x0, corr_y0, axw, axh])
            corr_axes.append(ax_c)

            if j >= len(chosen):
                ax_c.set_axis_off()
                continue

            cml_id, dist_m, r, df = chosen[j]
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()

            ax_c.scatter(x, y, s=10, alpha=0.65)

            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                m = float(np.nanmax(np.abs(np.concatenate([x[finite], y[finite]]))))
            else:
                m = 1.0
            if not np.isfinite(m) or m <= 0:
                m = 1.0

            ax_c.set_xlim(0.0, m)
            ax_c.set_ylim(0.0, m)
            ax_c.plot([0.0, m], [0.0, m], linewidth=1.0, alpha=0.9)

            ax_c.text(
                0.02,
                0.98,
                corr_title_str(int(cml_id), float(r)),
                transform=ax_c.transAxes,
                ha="left",
                va="top",
                fontsize=base_fs,
                linespacing=1.0,
            )

            ax_c.grid(True, alpha=0.3)
            ax_c.tick_params(axis="both", labelsize=base_fs, direction="out", pad=1.5)
            ax_c.set_xlabel("")
            ax_c.set_ylabel("")

        # ----------- shared labels (corr row) using supxlabel/supylabel -----------
        left = L
        right = L + k_links * axw + (k_links - 1) * hgap_use
        xmid = 0.5 * (left + right)
        ymid = corr_y0 + 0.5 * axh

        # X label: place relative to corr row (NOT figure bottom), then lock layout.
        sx = fig.supxlabel("1h úhrn stanice [mm]", fontsize=base_fs)
        sx.set_position((xmid, corr_y0 - 0.05))

        # Y label: place centered on corr row.
        sy = fig.supylabel("1h úhrn KMS [mm]", fontsize=base_fs, rotation=90)
        sy.set_position((left - 0.065, ymid))
        sy.set_in_layout(False)

        # ----------- save -----------
        st_name = station.get("name", "station")
        out_path = f"{out_dir}/{filename_prefix}_{st_name}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.1)

        saved.append(out_path)
        figs.append(fig)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figs, saved


def plot_stations_rolling1h_timeseries_only(
    *,
    calc_dataset: xr.Dataset,
    stations,
    start_dt: datetime,
    stop_dt: datetime,
    k_links: int = 4,
    tol: str = "6min",
    candidate_pool: int = 50,
    min_corr: float = 0.7,
    # skipping links
    skip_links_by_station=None,
    skip_links=None,
    # output
    out_dir: str = "./latex_figs",
    filename: str = "stations_rolling1h_timeseries.pdf",
    linewidth_pt: float = 483.69684,
    # styling / layout
    base_fs: float = 8.0,
    per_station_height_rel: float = 0.22,  # figure height per station (relative to width)
    legend_font_rel: int = 2,  # legend fontsize = base_fs - legend_font_rel
    show: bool = False,
):
    """
    One figure with subplots (one per station), showing ONLY rolling 1h sums time series:
    - station (CHMI SRA10M rolling 1h sum)
    - selected KMS links (TelcoRain R -> rolling 1h sum), selection MATCHES rolling1h_pdf logic
      (candidate_pool nearest-by-center, compute pearson on 1h sums, filter min_corr,
       choose closest by distance, fallback by (-r, distance)).
    No correlation scatter plots.

    This does NOT modify your existing plotting code.
    """
    import os

    import matplotlib.dates as mdates

    os.makedirs(out_dir, exist_ok=True)

    # TeX pt -> inch for matplotlib
    fig_width_in = float(linewidth_pt) / 72.27
    n = len(stations)
    fig_height_in = max(2.0, float(per_station_height_rel) * fig_width_in * n)

    start = to_rfc3339(start_dt)
    stop = to_rfc3339(stop_dt)

    global_skip = set(int(x) for x in (skip_links or []))

    # ---------- helpers copied from your rolling1h_pdf (selection must match) ----------
    def station_hour_sum_series(result_sra10m) -> pd.Series:
        times = [pd.to_datetime(p["time"]) for p in result_sra10m]
        vals = [(p["value"] or 0.0) for p in result_sra10m]
        s = pd.Series(vals, index=pd.DatetimeIndex(times)).sort_index()
        return s.rolling(window=6, min_periods=6).sum()

    def link_hour_sum_series(ds: xr.Dataset, cml_id: int) -> pd.Series:
        if "R_hour_sum" in ds:
            da = ds["R_hour_sum"].sel(cml_id=cml_id)
            return da.to_series().sort_index()
        da = ds["R"].sel(cml_id=cml_id).mean("channel_id", skipna=True)
        s = da.to_series().sort_index()
        s_10m = s * (10.0 / 60.0)
        return s_10m.rolling(window=6, min_periods=6).sum()

    def link_freq_ghz(ds: xr.Dataset, cml_id: int) -> float:
        if "frequency" not in ds:
            return np.nan
        try:
            v = np.asarray(ds["frequency"].sel(cml_id=cml_id).values, float).ravel()
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else np.nan
        except Exception:
            return np.nan

    def link_len_km(ds: xr.Dataset, cml_id: int) -> float:
        if "length" in ds:
            try:
                L = float(ds["length"].sel(cml_id=cml_id).values)
                return L if np.isfinite(L) else np.nan
            except Exception:
                pass
        a_lat = float(ds["site_a_latitude"].sel(cml_id=cml_id).values)
        a_lon = float(ds["site_a_longitude"].sel(cml_id=cml_id).values)
        b_lat = float(ds["site_b_latitude"].sel(cml_id=cml_id).values)
        b_lon = float(ds["site_b_longitude"].sel(cml_id=cml_id).values)
        return float(haversine_m(a_lat, a_lon, b_lat, b_lon)) / 1000.0

    def ts_legend_label(ds: xr.Dataset, cml_id: int, dist_m: float) -> str:
        d_km = float(dist_m) / 1000.0
        L = link_len_km(ds, cml_id)
        f = link_freq_ghz(ds, cml_id)

        ds_ = rf"$d={d_km:.1f}\,\mathrm{{km}}$"
        Ls = rf"$L={L:.1f}\,\mathrm{{km}}$" if np.isfinite(L) else r"$L=?$"
        fs = rf"$f={f:.1f}\,\mathrm{{GHz}}$" if np.isfinite(f) else r"$f=?$"
        return f"KMS {cml_id}, {ds_}, {Ls}, {fs}"

    # ---------- make figure ----------
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(fig_width_in, fig_height_in),
        sharex=True,
        constrained_layout=False,
    )
    if n == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        result = query_ws_data(start, stop, station["gh_id"])
        s_station = station_hour_sum_series(result["SRA10M"]).dropna()

        cands = nearest_links_by_center(
            calc_dataset,
            station,
            k=candidate_pool,
            require_rain=True,
        )

        # per-station skips (same behavior as your functions)
        station_skip = set()
        if skip_links_by_station:
            try:
                station_skip |= set(
                    int(x)
                    for x in (skip_links_by_station.get(station["gh_id"], []) or [])
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station["gh_id"], []) or []
                )
            try:
                station_skip |= set(
                    int(x)
                    for x in (skip_links_by_station.get(station.get("name"), []) or [])
                )
            except Exception:
                station_skip |= set(
                    skip_links_by_station.get(station.get("name"), []) or []
                )

        skip = global_skip | station_skip
        if skip:
            cands = [
                (cml_id, dist_m)
                for (cml_id, dist_m) in cands
                if int(cml_id) not in skip
            ]

        # evaluate correlations on 1h sums (selection logic match)
        evaluated = []
        for cml_id, dist_m in cands:
            cml_id = int(cml_id)
            if cml_id in skip:
                continue
            s_link = link_hour_sum_series(calc_dataset, cml_id).dropna()
            df = align_nearest(s_station, s_link, tol=tol)
            r = pearson_r(df["x"].to_numpy(), df["y"].to_numpy())
            if not np.isfinite(r):
                continue
            if min_corr is not None and r < float(min_corr):
                continue
            evaluated.append((cml_id, float(dist_m), float(r)))

        evaluated.sort(key=lambda x: x[1])  # by distance
        chosen_meta = evaluated[:k_links]

        if len(chosen_meta) < k_links:
            fallback = []
            chosen_ids = {c[0] for c in chosen_meta}
            for cml_id, dist_m in cands:
                cml_id = int(cml_id)
                if cml_id in skip or cml_id in chosen_ids:
                    continue
                s_link = link_hour_sum_series(calc_dataset, cml_id).dropna()
                df = align_nearest(s_station, s_link, tol=tol)
                r = pearson_r(df["x"].to_numpy(), df["y"].to_numpy())
                if not np.isfinite(r):
                    continue
                fallback.append((cml_id, float(dist_m), float(r)))
            fallback.sort(key=lambda x: (-x[2], x[1]))
            chosen_meta.extend(fallback[: (k_links - len(chosen_meta))])

        # ---- plot ----
        ax.plot(s_station.index, s_station.values, linewidth=1.2, label="Stanice")

        for cml_id, dist_m, r in chosen_meta:
            s_link = link_hour_sum_series(calc_dataset, int(cml_id)).dropna()
            ax.plot(
                s_link.index,
                s_link.values,
                linestyle="--",
                alpha=0.85,
                linewidth=1.0,
                label=ts_legend_label(calc_dataset, int(cml_id), float(dist_m)),
            )

        ax.set_title(station["name"], fontsize=base_fs + 1)
        ax.set_ylabel("1h úhrn [mm]", fontsize=base_fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=base_fs)

        ax.legend(
            loc="upper right",
            fontsize=max(6, base_fs - legend_font_rel),
            framealpha=0.9,
        )

    # shared X formatting (hours only, same as your other plots)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    axes[-1].set_xlabel("Čas [HH:MM]", fontsize=base_fs)

    # tighten vertical spacing a bit
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.10, hspace=0.25)

    out_path = f"{out_dir}/{filename}"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# PRAGUE 1

skip_links_by_station = {
    "P1PLIB01": {59},
}

figs, pdfs = plot_station_vs_links_corr_rolling1h_pdf(
    calc_dataset=calc_dataset_prague_1,
    stations=PRAGUE_STATIONS,
    start_dt=datetime(2025, 8, 29, 18, 0, tzinfo=timezone.utc),
    stop_dt=datetime(2025, 8, 30, 3, 0, tzinfo=timezone.utc),
    k_links=4,
    min_corr=0.7,
    out_dir="./latex_figs",
    filename_prefix="prague_event1_sum",
    linewidth_pt=483.69684,
    show=False,
    skip_links_by_station=skip_links_by_station,
)

figs, pdfs = plot_station_vs_links_corr(
    calc_dataset=calc_dataset_prague_1,
    stations=PRAGUE_STATIONS,
    start_dt=datetime(2025, 8, 29, 18, 0, tzinfo=timezone.utc),
    stop_dt=datetime(2025, 8, 30, 3, 0, tzinfo=timezone.utc),
    k_links=4,
    min_corr=0.7,
    out_dir="./latex_figs",
    filename_prefix="prague_event1_int",
    linewidth_pt=483.69684,
    show=False,
    skip_links_by_station=skip_links_by_station,
)


out_pdf = plot_stations_rolling1h_timeseries_only(
    calc_dataset=calc_dataset_prague_1,
    stations=PRAGUE_STATIONS,
    start_dt=datetime(2025, 8, 29, 18, 0, tzinfo=timezone.utc),
    stop_dt=datetime(2025, 8, 30, 3, 0, tzinfo=timezone.utc),
    k_links=4,
    min_corr=0.7,
    out_dir="./latex_figs",
    filename="prague_event1_sum_timeseries_only.pdf",
    skip_links_by_station=skip_links_by_station,
)

# PRAGUE 2

skip_links_by_station = {
    "P1PLIB01": {59},
    "P1PKLE01": {676, 240},
}


# figs, pdfs = plot_station_vs_links_corr_rolling1h_pdf(
#     calc_dataset=calc_dataset_prague_2,
#     stations=PRAGUE_STATIONS,
#     start_dt=datetime(2025, 9, 5, 18, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 9, 6, 8, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="prague_event2_sum",
#     linewidth_pt=483.69684,
#     show=False,
#     skip_links_by_station=skip_links_by_station,
# )

# figs, pdfs = plot_station_vs_links_corr(
#     calc_dataset=calc_dataset_prague_2,
#     stations=PRAGUE_STATIONS,
#     start_dt=datetime(2025, 9, 5, 18, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 9, 6, 8, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="prague_event2_int",
#     linewidth_pt=483.69684,
#     show=False,
#     skip_links_by_station=skip_links_by_station,
# )

# BRNO 1

# skip_links_by_station = {
#     "B2BZAB01": {936, 920},
# }

# figs, pdfs = plot_station_vs_links_corr_rolling1h_pdf(
#     calc_dataset=calc_dataset_brno_1,
#     stations=BRNO_STATIONS,
#     start_dt=datetime(2025, 7, 6, 16, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 7, 6, 23, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="brno_event1_sum",
#     linewidth_pt=483.69684,
#     show=False,
#     skip_links_by_station=skip_links_by_station,
# )

# figs, pdfs = plot_station_vs_links_corr(
#     calc_dataset=calc_dataset_brno_1,
#     stations=BRNO_STATIONS,
#     start_dt=datetime(2025, 7, 6, 16, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 7, 6, 23, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="brno_event1_int",
#     linewidth_pt=483.69684,
#     show=False,
#     skip_links_by_station=skip_links_by_station,
# )

# # BRNO 2

# figs, pdfs = plot_station_vs_links_corr_rolling1h_pdf(
#     calc_dataset=calc_dataset_brno_2,
#     stations=BRNO_STATIONS,
#     start_dt=datetime(2025, 9, 10, 10, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 9, 11, 3, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="brno_event2_sum",
#     linewidth_pt=483.69684,
#     show=False,
#     # skip_links_by_station=skip_links_by_station,
# )

# figs, pdfs = plot_station_vs_links_corr(
#     calc_dataset=calc_dataset_brno_2,
#     stations=BRNO_STATIONS,
#     start_dt=datetime(2025, 9, 10, 10, 0, tzinfo=timezone.utc),
#     stop_dt=datetime(2025, 9, 11, 3, 0, tzinfo=timezone.utc),
#     k_links=4,
#     min_corr=0.7,
#     out_dir="./latex_figs",
#     filename_prefix="brno_event2_int",
#     linewidth_pt=483.69684,
#     show=False,
#     # skip_links_by_station=skip_links_by_station,
# )
