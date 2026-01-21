from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------
# Config
# --------------------------

CSV_PATH = "export.csv"
OUT_PNG = "links_cz_basemap.png"

# tvoje názvy sloupců
LON_A = "Zeměpisná délka A"
LON_B = "Zeměpisná délka B"
LAT_A = "Zeměpisná šířka A"
LAT_B = "Zeměpisná šířka B"
DIST_KM = "Vzdálenost"  # km

# ---- filtr podle délky (km) ----
MAX_LEN_KM: float | None = 10
MIN_LEN_KM: float | None = None

# Přibližný bbox ČR (WGS84)
USE_FIXED_CZ_BOUNDS = True
CZ_BOUNDS_WGS84 = [
    [48.55, 12.05],  # [lat_min, lon_min]
    [51.10, 18.90],  # [lat_max, lon_max]
]

# Basemap
BASEMAP_ZOOM = 8
PAD_M = 8_000.0  # menší padding => méně okolních států (dřív 15 km)

# --- Crop bboxu (v EPSG:3857) ---
CROP_MODE = "center"  # "none" | "center" | "point"
CROP_SCALE = (
    1  # <1 => víc “utáhnout” na ČR (zkus 0.90 nebo 0.88, pokud chceš ještě víc)
)

# Styling
LINEWIDTH = 1.0
LINE_ALPHA = 1.0
LINE_COLOR = "black"

DPI = 220


# --------------------------
# Helpers
# --------------------------


def _crop_bounds_3857(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    scale: float,
    center_xy=None,
):
    if not (scale > 0):
        raise ValueError("scale must be > 0")

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    if center_xy is not None:
        cx, cy = center_xy

    w = (xmax - xmin) * scale
    h = (ymax - ymin) * scale
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def filter_links_by_length(
    gdf, *, dist_col: str, max_km: float | None = None, min_km: float | None = None
):
    if dist_col not in gdf.columns:
        raise ValueError(
            f"Sloupec '{dist_col}' v datech neexistuje, nelze filtrovat podle délky."
        )

    d = pd.to_numeric(gdf[dist_col], errors="coerce")
    mask = d.notna()

    if max_km is not None:
        mask &= d <= float(max_km)
    if min_km is not None:
        mask &= d >= float(min_km)

    return gdf.loc[mask].copy()


def load_links_gdf_from_csv(csv_path: str):
    import geopandas as gpd
    from shapely.geometry import LineString

    df = pd.read_csv(csv_path)

    for c in [LON_A, LON_B, LAT_A, LAT_B, DIST_KM]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    m = (
        np.isfinite(df[LON_A])
        & np.isfinite(df[LON_B])
        & np.isfinite(df[LAT_A])
        & np.isfinite(df[LAT_B])
    )
    df = df.loc[m].copy()

    geoms = []
    keep = []
    for i, r in df.iterrows():
        lon1, lat1, lon2, lat2 = (
            float(r[LON_A]),
            float(r[LAT_A]),
            float(r[LON_B]),
            float(r[LAT_B]),
        )

        if not (
            -180 <= lon1 <= 180
            and -180 <= lon2 <= 180
            and -90 <= lat1 <= 90
            and -90 <= lat2 <= 90
        ):
            continue

        geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
        keep.append(i)

    if not geoms:
        raise ValueError("No valid link geometries found in CSV.")

    df2 = df.loc[keep].copy()
    return gpd.GeoDataFrame(df2, geometry=geoms, crs="EPSG:4326")


def prepare_base_map_from_gdf(
    gdf_wgs84,
    *,
    bounds_wgs84=None,
    zoom: int = 8,
    pad_m: float = 0.0,
    crop_mode: str = "none",
    crop_scale: float = 1.0,
):
    import contextily as ctx
    from pyproj import Transformer

    gdf_3857 = gdf_wgs84.to_crs(epsg=3857)

    if bounds_wgs84 is None:
        xmin, ymin, xmax, ymax = gdf_3857.total_bounds
        pad_auto = max(2000.0, 0.05 * max(xmax - xmin, ymax - ymin))
        xmin -= pad_auto + pad_m
        xmax += pad_auto + pad_m
        ymin -= pad_auto + pad_m
        ymax += pad_auto + pad_m
    else:
        (lat_min, lon_min), (lat_max, lon_max) = bounds_wgs84
        tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        corners_lonlat = [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_min),
            (lon_max, lat_max),
        ]
        xs, ys = zip(*(tf.transform(lon, lat) for lon, lat in corners_lonlat))
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))

        xmin -= pad_m
        xmax += pad_m
        ymin -= pad_m
        ymax += pad_m

    crop_mode = (crop_mode or "none").lower().strip()
    if crop_mode != "none":
        if crop_mode == "center":
            xmin, ymin, xmax, ymax = _crop_bounds_3857(
                xmin, ymin, xmax, ymax, scale=crop_scale
            )
        else:
            raise ValueError("crop_mode supported here: 'none' | 'center'")

    img, extent = ctx.bounds2img(
        xmin,
        ymin,
        xmax,
        ymax,
        zoom=zoom,
        source=ctx.providers.OpenStreetMap.Mapnik,
        ll=False,
    )

    return {
        "gdf_3857": gdf_3857,
        "basemap_img": img,
        "basemap_extent": extent,
        "map_bounds_3857": (xmin, ymin, xmax, ymax),
    }


def save_links_map(prepared, *, out_png: str, dpi: int = 200, title_extra: str = ""):
    import matplotlib.pyplot as plt

    basemap_img = prepared["basemap_img"]
    W, E, S, N = prepared["basemap_extent"]
    gdf = prepared["gdf_3857"]

    width_m = E - W
    height_m = N - S
    aspect = width_m / height_m

    fig_h = 8.0
    fig_w = fig_h * aspect

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(basemap_img, extent=[W, E, S, N], origin="upper", zorder=0)

    # ✅ černé spoje
    gdf.plot(
        ax=ax,
        linewidth=LINEWIDTH,
        alpha=LINE_ALPHA,
        color=LINE_COLOR,
        zorder=10,
    )

    ax.set_xlim(W, E)
    ax.set_ylim(S, N)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.savefig(out_png, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return out_png


def main() -> int:
    gdf = load_links_gdf_from_csv(CSV_PATH)

    title_extra = ""
    if MAX_LEN_KM is not None or MIN_LEN_KM is not None:
        gdf = filter_links_by_length(
            gdf, dist_col=DIST_KM, max_km=MAX_LEN_KM, min_km=MIN_LEN_KM
        )
        if MAX_LEN_KM is not None and MIN_LEN_KM is None:
            title_extra = f" (≤ {MAX_LEN_KM:g} km)"
        elif MIN_LEN_KM is not None and MAX_LEN_KM is None:
            title_extra = f" (≥ {MIN_LEN_KM:g} km)"
        elif MIN_LEN_KM is not None and MAX_LEN_KM is not None:
            title_extra = f" ({MIN_LEN_KM:g}–{MAX_LEN_KM:g} km)"

    if len(gdf) == 0:
        raise RuntimeError("Po aplikaci filtru nezbyl žádný spoj k vykreslení.")

    bounds = CZ_BOUNDS_WGS84 if USE_FIXED_CZ_BOUNDS else None

    prepared = prepare_base_map_from_gdf(
        gdf,
        bounds_wgs84=bounds,
        zoom=BASEMAP_ZOOM,
        pad_m=PAD_M,
        crop_mode=CROP_MODE,
        crop_scale=CROP_SCALE,
    )

    out = save_links_map(prepared, out_png=OUT_PNG, dpi=DPI, title_extra=title_extra)
    print("Saved:", out)
    print("Links plotted:", len(prepared["gdf_3857"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
