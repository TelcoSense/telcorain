#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Side-by-side comparison grid:
- LEFT: TelcoRain overlays (YYYY-MM-DD_HHMM[_score].png)
- RIGHT: CHMI radar MAXZ overlays (T_..._YYYYMMDDHHMMSS.png)

Produces a single PNG with N rows × 2 columns (no empty slots), same basemap & extent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import xarray as xr

# --------------------------
# Config (EDIT THESE)
# --------------------------

FOLDER = "./case_study_runs/prague_1"
CALC_DATASET_NC = f"{FOLDER}/outputs_web/calc_dataset.nc"

# TelcoRain overlays are in outputs_web/*.png (your format)
TELCORAIN_OVERLAYS_DIR = f"{FOLDER}/outputs_web"

# CHMI MAXZ overlays are in maxz/*.png (your format)
MAXZ_OVERLAYS_DIR = f"{FOLDER}/maxz"

# Overlay geographic bounds (WGS84 lat/lon) - you already have these
OVERLAY_BOUNDS = [
    [48.047, 11.267],
    [51.458, 19.624],
]

# Time range for selection
# START = datetime(2025, 7, 6, 19, 20)
# STOP = datetime(2025, 7, 6, 21, 20)

START = datetime(2025, 8, 29, 22, 0)
STOP = datetime(2025, 8, 29, 23, 0)

# Pairing tolerance (TelcoRain timestamp matched to nearest MAXZ timestamp)
PAIR_TOL = timedelta(minutes=6)

# Output image
OUT_PNG = "telcorain_vs_maxz.png"

# Basemap zoom (contextily)
BASEMAP_ZOOM = 11

# Rendering
FIG_HEIGHT = 10.0
DPI = 140
OVERLAY_ALPHA_LEFT = 0.40  # TelcoRain
OVERLAY_ALPHA_RIGHT = 0.4  # MAXZ
DRAW_LINKS = False
LINEWIDTH = 1.4

CROP_MODE = "center"  # "none" | "center" | "point"
CROP_SCALE = 0.70  # 0.70 means keep 70% of width+height around the chosen center

# if you want “point crop” (explicit center in lat/lon), set:
CROP_CENTER_LATLON = (50.0755, 14.4378)


# --------------------------
# Basemap preparation
# --------------------------


def _crop_bounds_3857(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    scale: float,
    center_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Crop (scale < 1) or expand (scale > 1) a 3857 bbox around its center or a given center.
    """
    if not (scale > 0):
        raise ValueError("scale must be > 0")

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    if center_xy is not None:
        cx, cy = center_xy

    w = (xmax - xmin) * scale
    h = (ymax - ymin) * scale

    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def _latlon_to_3857(lat: float, lon: float) -> Tuple[float, float]:
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


# 3) Now replace your prepare_base_map() with this version (same signature, extra crop args)


def prepare_base_map(
    ds: xr.Dataset,
    *,
    bounds=None,  # [[lat_min, lon_min], [lat_max, lon_max]] or None
    zoom: int = 12,
    pad_m: float = 0.0,
    crop_mode: str = "none",  # "none" | "center" | "point"
    crop_scale: float = 1.0,  # <1 => crop in, >1 => expand
    crop_center_latlon: Optional[Tuple[float, float]] = None,  # (lat, lon)
) -> Dict:
    import contextily as ctx
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString

    a_lat = np.asarray(ds["site_a_latitude"].values, float)
    a_lon = np.asarray(ds["site_a_longitude"].values, float)
    b_lat = np.asarray(ds["site_b_latitude"].values, float)
    b_lon = np.asarray(ds["site_b_longitude"].values, float)

    lines = []
    for i in range(len(a_lat)):
        if not (
            np.isfinite(a_lat[i])
            and np.isfinite(a_lon[i])
            and np.isfinite(b_lat[i])
            and np.isfinite(b_lon[i])
        ):
            continue
        lines.append(
            LineString(
                [(float(a_lon[i]), float(a_lat[i])), (float(b_lon[i]), float(b_lat[i]))]
            )
        )

    if not lines:
        raise ValueError("No valid link geometries found (site_* lat/lon missing?).")

    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_crs(epsg=3857)

    # base bbox
    if bounds is None:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        pad = max(2000.0, 0.05 * max(xmax - xmin, ymax - ymin))
        xmin -= pad + pad_m
        xmax += pad + pad_m
        ymin -= pad + pad_m
        ymax += pad + pad_m
    else:
        (lat_min, lon_min), (lat_max, lon_max) = bounds
        bbox = gpd.GeoSeries.from_bbox(
            (lon_min, lat_min, lon_max, lat_max),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        xmin, ymin, xmax, ymax = bbox.total_bounds
        xmin -= pad_m
        xmax += pad_m
        ymin -= pad_m
        ymax += pad_m

    # optional crop/expand
    crop_mode = (crop_mode or "none").lower().strip()
    if crop_mode != "none":
        if crop_mode == "center":
            xmin, ymin, xmax, ymax = _crop_bounds_3857(
                xmin, ymin, xmax, ymax, scale=crop_scale, center_xy=None
            )
        elif crop_mode == "point":
            if crop_center_latlon is None:
                raise ValueError(
                    "crop_mode='point' requires crop_center_latlon=(lat,lon)"
                )
            cx, cy = _latlon_to_3857(crop_center_latlon[0], crop_center_latlon[1])
            xmin, ymin, xmax, ymax = _crop_bounds_3857(
                xmin, ymin, xmax, ymax, scale=crop_scale, center_xy=(cx, cy)
            )
        else:
            raise ValueError("crop_mode must be one of: 'none', 'center', 'point'")

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
        "gdf_3857": gdf,
        "basemap_img": img,
        "basemap_extent": extent,
        "map_bounds_3857": (xmin, ymin, xmax, ymax),
    }


# --------------------------
# Overlay list building
# --------------------------

_TELCORAIN_RE = re.compile(
    r"""
    (?P<date>\d{4}-\d{2}-\d{2})
    _
    (?P<hhmm>\d{4})
    (?:_(?P<score>\d+(?:\.\d+)?))?
    \.png$
    """,
    re.VERBOSE,
)

_MAXZ_RE = re.compile(
    r"""
    ^T_[A-Z0-9]+_C_[A-Z0-9]+_
    (?P<ts>\d{14})
    \.(?:png|PNG)$
    """,
    re.VERBOSE,
)


def build_overlays_list_telcorain(
    folder: str,
    *,
    start: Optional[datetime] = None,
    stop: Optional[datetime] = None,
    min_score: Optional[float] = None,
    label_fmt: str = "%Y-%m-%d %H:%M",
) -> List[dict]:
    overlays: List[dict] = []
    for fn in os.listdir(folder):
        m = _TELCORAIN_RE.match(fn)
        if not m:
            continue

        ts = datetime.strptime(f"{m.group('date')} {m.group('hhmm')}", "%Y-%m-%d %H%M")

        if start and ts < start:
            continue
        if stop and ts > stop:
            continue

        score = m.group("score")
        score_f = float(score) if score is not None else None

        if min_score is not None and score_f is not None and score_f < min_score:
            continue

        overlays.append(
            {
                "path": os.path.join(folder, fn),
                "label": ts.strftime(label_fmt),
                "ts": ts,
                "score": score_f,
                "fn": fn,
            }
        )

    overlays.sort(key=lambda x: x["ts"])
    return overlays


def build_overlays_list_maxz(
    folder: str,
    *,
    start: Optional[datetime] = None,
    stop: Optional[datetime] = None,
    label_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> List[dict]:
    overlays: List[dict] = []
    for fn in os.listdir(folder):
        m = _MAXZ_RE.match(fn)
        if not m:
            continue

        ts = datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S")

        if start and ts < start:
            continue
        if stop and ts > stop:
            continue

        overlays.append(
            {
                "path": os.path.join(folder, fn),
                "label": ts.strftime(label_fmt),
                "ts": ts,
                "score": None,
                "fn": fn,
            }
        )

    overlays.sort(key=lambda x: x["ts"])
    return overlays


# --------------------------
# Pairing
# --------------------------


def pair_overlays_by_time(
    left: List[dict],
    right: List[dict],
    *,
    tol: timedelta = timedelta(minutes=5),
) -> List[Tuple[dict, Optional[dict]]]:
    """
    For each left overlay (TelcoRain), find nearest right overlay (MAXZ) within tol.
    Returns list of (left, matched_right_or_None).
    """
    if not left:
        return []

    right_sorted = sorted(right, key=lambda x: x["ts"])
    out: List[Tuple[dict, Optional[dict]]] = []

    r_idx = 0
    for L in left:
        t = L["ts"]
        if not right_sorted:
            out.append((L, None))
            continue

        while r_idx + 1 < len(right_sorted) and abs(
            right_sorted[r_idx + 1]["ts"] - t
        ) <= abs(right_sorted[r_idx]["ts"] - t):
            r_idx += 1

        R = right_sorted[r_idx]
        if abs(R["ts"] - t) <= tol:
            out.append((L, R))
        else:
            out.append((L, None))

    return out


# --------------------------
# Rendering
# --------------------------


def save_overlays_side_by_side(
    prepared: Dict,
    *,
    pairs: List[Tuple[dict, Optional[dict]]],
    overlay_bounds,
    out_png: str,
    left_title: str = "TelcoRain (calc)",
    right_title: str = "CHMI radar MAXZ",
    fig_height: float = 10.0,
    dpi: int = 140,
    overlay_alpha_left: float = 0.40,
    overlay_alpha_right: float = 0.55,
    draw_links: bool = False,
    linewidth: float = 1.4,
    wspace: float = 0.006,
    hspace: float = 0.012,
    title_fontsize: int = 11,
    rowlabel_fontsize: int = 9,
    missing_text: str = "(missing)",
) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pyproj import Transformer

    if not pairs:
        raise ValueError("No overlay pairs to plot.")

    basemap_img = prepared["basemap_img"]
    W, E, S, N = prepared["basemap_extent"]  # [W,E,S,N] in EPSG:3857
    gdf = prepared["gdf_3857"]

    width_m = E - W
    height_m = N - S
    panel_aspect = width_m / height_m  # width/height

    nrows = len(pairs)

    row_h = fig_height / nrows
    panel_w = row_h * panel_aspect
    fig_w = 2.0 * panel_w
    fig_h = fig_height

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = GridSpec(
        nrows=nrows,
        ncols=2,
        figure=fig,
        wspace=wspace,
        hspace=hspace,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )

    (olat_min, olon_min), (olat_max, olon_max) = overlay_bounds
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ox0, oy0 = tf.transform(olon_min, olat_min)
    ox1, oy1 = tf.transform(olon_max, olat_max)

    # Column headers
    fig.text(
        0.25,
        0.995,
        left_title,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        color="black",
        bbox=dict(
            facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.25"
        ),
        zorder=200,
    )
    fig.text(
        0.75,
        0.995,
        right_title,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        color="black",
        bbox=dict(
            facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.25"
        ),
        zorder=200,
    )

    # Separator line
    fig.lines.append(
        plt.Line2D(
            [0.5, 0.5],
            [0.0, 1.0],
            transform=fig.transFigure,
            linewidth=1.0,
            alpha=0.18,
            color="black",
        )
    )

    for r, (L, R) in enumerate(pairs):
        for c in (0, 1):
            ax = fig.add_subplot(gs[r, c])

            ax.imshow(basemap_img, extent=[W, E, S, N], origin="upper", zorder=0)

            ov = L if c == 0 else R
            if ov is not None and ov.get("path") and os.path.exists(ov["path"]):
                img = plt.imread(ov["path"])
                ax.imshow(
                    img,
                    extent=[ox0, ox1, oy0, oy1],
                    origin="upper",
                    alpha=(overlay_alpha_left if c == 0 else overlay_alpha_right),
                    zorder=5,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    missing_text,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=title_fontsize,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                        boxstyle="round,pad=0.3",
                    ),
                    zorder=50,
                )

            if draw_links:
                gdf.plot(ax=ax, linewidth=linewidth, alpha=0.95, zorder=10)

            ax.set_xlim(W, E)
            ax.set_ylim(S, N)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()

        # Row label (timestamp of the LEFT image)
        row_label = L.get("label") or L["ts"].strftime("%Y-%m-%d %H:%M")
        fig.text(
            0.5,
            1.0 - (r + 0.5) / nrows,
            row_label,
            ha="center",
            va="center",
            fontsize=rowlabel_fontsize,
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.45,
                edgecolor="none",
                boxstyle="round,pad=0.18",
            ),
            zorder=150,
        )

    fig.savefig(out_png, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return out_png


# --------------------------
# Main
# --------------------------


def main() -> int:
    if not os.path.exists(CALC_DATASET_NC):
        raise FileNotFoundError(f"calc_dataset.nc not found: {CALC_DATASET_NC}")

    if not os.path.isdir(TELCORAIN_OVERLAYS_DIR):
        raise FileNotFoundError(
            f"TelcoRain overlays dir not found: {TELCORAIN_OVERLAYS_DIR}"
        )

    if not os.path.isdir(MAXZ_OVERLAYS_DIR):
        raise FileNotFoundError(f"MAXZ overlays dir not found: {MAXZ_OVERLAYS_DIR}")

    ds = xr.open_dataset(CALC_DATASET_NC)

    # prepared = prepare_base_map(ds, zoom=BASEMAP_ZOOM)

    prepared = prepare_base_map(
        ds,
        zoom=BASEMAP_ZOOM,
        crop_mode=CROP_MODE,
        crop_scale=CROP_SCALE,
        crop_center_latlon=CROP_CENTER_LATLON,
    )

    left = build_overlays_list_telcorain(
        TELCORAIN_OVERLAYS_DIR,
        start=START,
        stop=STOP,
        label_fmt="%Y-%m-%d %H:%M",
    )

    right = build_overlays_list_maxz(
        MAXZ_OVERLAYS_DIR,
        start=START,
        stop=STOP,
        label_fmt="%Y-%m-%d %H:%M:%S",
    )

    if not left:
        raise RuntimeError("No TelcoRain overlays found in the selected time range.")
    if not right:
        raise RuntimeError("No MAXZ overlays found in the selected time range.")

    pairs = pair_overlays_by_time(left, right, tol=PAIR_TOL)

    out = save_overlays_side_by_side(
        prepared,
        pairs=pairs,
        overlay_bounds=OVERLAY_BOUNDS,
        out_png=OUT_PNG,
        left_title="TelcoRain (calc)",
        right_title="CHMI radar MAXZ",
        fig_height=FIG_HEIGHT,
        dpi=DPI,
        overlay_alpha_left=OVERLAY_ALPHA_LEFT,
        overlay_alpha_right=OVERLAY_ALPHA_RIGHT,
        draw_links=DRAW_LINKS,
        linewidth=LINEWIDTH,
    )

    print(f"Saved: {out}")
    print(f"Rows: {len(pairs)} (left images), pairing tol: {PAIR_TOL}")
    missing = sum(1 for _, r in pairs if r is None)
    if missing:
        print(f"Warning: {missing} rows have missing MAXZ match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
