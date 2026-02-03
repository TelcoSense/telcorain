"""Utilities for running the *custom* wet/dry CNN trained on 30 s TRSL and
mapping its output to the main pipeline time step.

Motivation
----------
The Kaleta ("ours") CNN is trained on 30 s sampled TRSL with a sample_size of 60,
producing one wet probability per 30 min segment. TelcoRain typically operates
with 10 min time steps for rain-rate mapping; therefore, the CNN output must be
computed on 30 s data and then aggregated/reindexed to 10 min.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from telcorain.procedures.wet_dry import preprocess_utility
from telcorain.procedures.wet_dry.cnn_utility import cnn_infer_only


def _preprocess_keep_time(
    ds_30s: xr.Dataset,
    *,
    interp_max_gap: int = 10,
) -> pd.DataFrame:
    """A thin wrapper around the existing preprocessing that *keeps* time stamps.

    The original cml_preprocess drops rows and resets index. That breaks
    alignment when attaching outputs back to the xarray timeline.

    Here we follow the same computations but keep the original `time` column.
    """

    df = preprocess_utility.convert_xarray_to_cml_df(ds_30s)
    df["trsl_A"] = df["tsl_A"] - df["rsl_A"]
    df["trsl_B"] = df["tsl_B"] - df["rsl_B"]

    df["trsl_A"] = df["trsl_A"].where(df["trsl_A"] < 99.0)
    df["trsl_B"] = df["trsl_B"].where(df["trsl_B"] < 99.0)

    # interpolate numeric columns (limited) but keep row count
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        limit=interp_max_gap, method="linear"
    )

    # Apply the same optional steps as used in TelcoRain's current inference.
    # (These mirror the parameters passed in rain_calculation.py)
    df = preprocess_utility.cml_suppress_extremes_std(df, window_size=10, std_threshold=5.0)
    df = preprocess_utility.cml_suppress_extremes_z(df, z_threshold=10.0)
    df = preprocess_utility.cml_temp_extremes_std(df)
    df = preprocess_utility.subtract_trsl_median(df)

    # Standardisation (same as cml_preprocess)
    for trsl in ["trsl_A", "trsl_B"]:
        cml_mean = df[trsl].mean()
        cml_max = df[trsl].max()
        if pd.isna(cml_max) or cml_max == 0:
            df[trsl] = np.nan
        else:
            df[trsl] = (df[trsl].values - cml_mean) / cml_max

    for temp in ["temperature_rx", "temperature_tx"]:
        temp_min = df[temp].min()
        temp_max = df[temp].max()
        if pd.isna(temp_min) or pd.isna(temp_max) or temp_max == temp_min:
            df[temp] = np.nan
        else:
            df[temp] = (df[temp].values - temp_min) / (temp_max - temp_min)

    return df


def compute_wet_mask_10min_from_30s(
    ds_30s: xr.Dataset,
    target_time: xr.DataArray,
    *,
    model_param_dir: str,
    sample_size: int = 60,
    threshold: float = 0.5,
    target_rule: str = "max",
    fillna_dry: bool = True,
) -> np.ndarray:
    """Run the custom CNN on 30 s data and map the result to target timestamps.

    Parameters
    ----------
    ds_30s:
        Link dataset at ~30 s sampling (dims: channel_id,time).
    target_time:
        Time coordinate of the main pipeline dataset (typically 10 min sampling).
    model_param_dir:
        Path/name under cnn_custom_model/ with the saved torch state dict.
    sample_size:
        Number of 30 s samples per CNN segment (60 => 30 min).
    threshold:
        Wet/dry threshold on the CNN probability.
    target_rule:
        Aggregation within each 10 min bin: "max" (any wet) or "mean".
    fillna_dry:
        If True, NaNs after reindexing are treated as dry (0).

    Returns
    -------
    wet_10min : np.ndarray
        Array aligned to `target_time` with values 0/1.
    """

    df = _preprocess_keep_time(ds_30s)

    # Infer probabilities per 30 min segment.
    probs = cnn_infer_only(
        preprocessed_df=df,
        param_dir=model_param_dir,
        num_channels=2,
        sample_size=sample_size,
        batchsize=256,
        single_output=True,
    )

    # Expand segment probabilities back to per-30s timeline.
    n = len(df)
    n_segments = len(probs)
    expanded = np.full(n, np.nan, dtype=float)
    max_fill = min(n, n_segments * sample_size)
    if max_fill > 0:
        expanded[:max_fill] = np.repeat(probs, sample_size)[:max_fill]

    # Convert to binary wet flags on the 30 s grid.
    wet_30s = np.where(np.isnan(expanded), np.nan, (expanded >= threshold).astype(int))

    # Aggregate to 10 min.
    s = pd.Series(wet_30s, index=pd.to_datetime(df["time"], utc=True))
    # Force 10 minute boundaries; rule uses integer 0/1 so max=any wet.
    if target_rule == "mean":
        s10 = s.resample("10min").mean()
        s10 = (s10 >= 0.5).astype(float)
    else:
        s10 = s.resample("10min").max()

    # Reindex to target_time exactly.
    tgt_idx = pd.to_datetime(target_time.values, utc=True)
    s10 = s10.reindex(tgt_idx)
    if fillna_dry:
        s10 = s10.fillna(0.0)

    return s10.astype(int).to_numpy()
