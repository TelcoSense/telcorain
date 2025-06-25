#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filename: preprocess_utility.py
Author: Lukas Kaleta
Date: 2025-05-26
Version: 2.0
Description:
    This script contains functions for cml dataset preprocessing
    Specifically for Czech CML dataset

License:
Contact: 211312@vutbr.cz
"""


""" Imports """
# Import python libraries

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import find_peaks

# Import external packages


# Import local modules

""" Notes """
# TODO: copy data into NaN gaps from adjacent cml #cml['rsl_A'] = cml.rsl_A + cml.rsl_B.where(np.isnan(cml.rsl_A))
# TODO: copy adjacent data, if large chunk of rsl is missing: alpha: 12
# TODO: better interpolation: https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe
# TODO: some CML still has NaN gaps, find it and exclude the NaN samples.
# TODO: Different preprocess tresholds for different cml technologies
# TODO: Data augmentation: noise injecting, time warp, Random scaling, mixUp/cutMix
# TODO: patch the Ref WD calculation: its downsampled, but the samples are deleted in CML preprocessing, WD samples are not aligned


""" Variable definitions """


""" Function definitions """


def convert_xarray_to_cml_df(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert xarray.Dataset with dims (channel_id, time) into a flat DataFrame
    suitable for cml_preprocess. Assumes channel_id[0] = A, channel_id[1] = B.
    """
    # Extract channel indices
    if ds.channel_id.size != 2:
        raise ValueError("Expected exactly 2 channels: A and B")

    df = pd.DataFrame(
        {
            "time": ds.time.values,
            "tsl_A": ds.tsl.values[0],
            "tsl_B": ds.tsl.values[1],
            "rsl_A": ds.rsl.values[0],
            "rsl_B": ds.rsl.values[1],
            "temperature_rx": ds.temperature_rx.values[0],
            "temperature_tx": ds.temperature_tx.values[1],
        }
    )

    return df


def cml_preprocess(
    cml: pd.DataFrame,
    interp_max_gap=10,
    suppress_step=False,
    conv_threshold=20.0,
    std_method=False,
    window_size=10,
    std_threshold=5.0,
    z_method=False,
    z_threshold=10.0,
    reset_detect=False,
    temp_extremes=True,
    subtract_median=True,
):
    """
    Preprocess cml dataset: Interpolate gaps, Exclude NaN values,
    standardise values by subtracting mean and scaling to 0-1.
    Optional:
        Remove fault extreme values in trsl series using STD or Z method,
        Detect steps in trsl mean value and alighn periods with extremely different mean value,
        Detect cml reset using uptime
        Suppres fault extreme values and non-smooth value steps in cml temperature

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    interp_max_gap : int, default value = 10, maximal gap in each data column to be interpolated.
    suppress_step : boolean, default = False, perform trsl step compensation if True.
    conv_threshold : float, default = 20.0, threshold for Large steps detection using convolution.
        Adjust this based on normal data fluctuations
    std_method : boolean, default = False, perform trsl extreme detection using std method if True.
    window_size : int, default = 10, Window size for rolling STD. Adjust based on your data characteristics.
    std_threshold : float, default = 5.0, threshold for extreme value detection using STD.
        Adjust this based on normal data fluctuations
    z_method : boolean, default = False, perform trsl extreme detection using Z method if True.
    z_threshold : float, default = 10.0, threshold for extreme value detection using Z method.
        Adjust this based on fluctuations of your data
    reset_detect : boolean, default = True, perform trsl reset detection if True.
    temp_extremes : boolean, default = True, perform temperature smoothing if True.
    subtract_median : boolean, default = True, subtract rolling median if True.

    Returns
    cml : Pandas.DataFrame
    """

    cml = convert_xarray_to_cml_df(cml)

    # modify it for our data
    cml["trsl_A"] = cml.tsl_A - cml.rsl_A
    cml["trsl_B"] = cml.tsl_B - cml.rsl_B

    # exclude extreme values
    cml["trsl_A"] = cml.trsl_A.where(cml.trsl_A < 99.0)
    cml["trsl_B"] = cml.trsl_B.where(cml.trsl_B < 99.0)

    # First interpolation both trsl, and R and drop missing values
    # cml = cml.interpolate(axis=0, method="linear", limit=interp_max_gap)
    # cml = cml.dropna(axis=0, how="all", subset=["trsl_A", "trsl_B"])

    numeric_cols = cml.select_dtypes(include=["number"]).columns
    cml[numeric_cols] = cml[numeric_cols].interpolate(
        limit=interp_max_gap, method="linear"
    )
    cml = cml.dropna(subset=["trsl_A", "trsl_B"], how="all").reset_index(drop=True)

    # Anomaly handling
    # if reset_detect:
    #     cml = cml_reset_detect(cml)
    if suppress_step:
        cml = cml_suppress_step(cml, conv_threshold)
    if std_method:
        cml = cml_suppress_extremes_std(cml, window_size, std_threshold)
    if z_method:
        cml = cml_suppress_extremes_z(cml, z_threshold)
    if temp_extremes:
        cml = cml_temp_extremes_std(cml)
    if subtract_median:
        cml = subtract_trsl_median(cml)

    # standardisation
    for trsl in ["trsl_A", "trsl_B"]:
        # trsl: MEAN-MAX standardization
        cml_mean = cml[trsl].mean()
        cml_max = cml[trsl].max()
        cml[trsl] = (cml[trsl].values - cml_mean) / cml_max
    for temp in ["temperature_rx", "temperature_tx"]:
        # temperature: MIN-MAX standardization
        temp_min = cml[temp].min()
        temp_max = cml[temp].max()
        if temp_max != temp_min:
            cml[temp] = (cml[temp].values - temp_min) / (temp_max - temp_min)

    return cml


def cml_suppress_extremes_std(cml: pd.DataFrame, window_size=10, std_threshold=5.0):
    """
    Remove fault extreme values in trsl series by calculating floating window std,
    interpolate missing values

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    window_size : int, default = 10, Window size for rolling STD. Adjust based on your data characteristics.
    std_threshold : float, default = 5.0, threshold for Large steps and extreme value detection using STD.
        Adjust this based on normal data fluctuations

    Returns
    cml : Pandas.DataFrame
    """
    # calculate rolling STD
    for trsl in ["trsl_A", "trsl_B"]:
        rolling_std = cml[trsl].rolling(window=window_size, center=True).std()

        # Fill NaN values at the edges
        rolling_std.fillna(method="bfill", inplace=True)
        rolling_std.fillna(method="ffill", inplace=True)

        # drop values with STD above the threshold
        cml[trsl] = cml[trsl].where(np.abs(rolling_std) < std_threshold)

    # interpolation both trsl, and R
    # cml = cml.interpolate(axis=0, method="linear")

    return cml


def cml_suppress_extremes_z(cml: pd.DataFrame, z_threshold=10.0):
    """
    Remove fault extreme values in trsl series based on Z method:
    Z = (x-mean)/std
    interpolate missing values

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    z_threshold : float, default = 10.0, threshold for extreme value detection using Z method.
        Adjust this based on fluctuations of your data

    Returns
    cml : Pandas.DataFrame
    """
    for trsl in ["trsl_A", "trsl_B"]:
        # Drop faulty single extreme values by Z method (non detected by std)
        z_param = (cml[trsl] - cml[trsl].mean()) / cml[trsl].std()
        cml[trsl] = cml[trsl].where(abs(z_param) < z_threshold)
        cml[trsl] = cml[trsl].where(z_param > -5.0)
    # interpolation both trsl, and R
    # cml = cml.interpolate(axis=0, method="linear")

    return cml


def cml_suppress_step(cml: pd.DataFrame, conv_threshold=250.0):
    """
    Detect steps in trsl mean value and alighn periods with extremely different mean value

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.
    conv_threshold : float, default = 20.0, Threshold for detecting step from convolution result
        pick values between (10;50)

    Returns
    cml : Pandas.DataFrame
    """
    # create the step function
    step = np.hstack((np.ones(500), -1 * np.ones(500)))

    for col_name in ["trsl_A", "trsl_B"]:
        trsl = cml[col_name].copy()

        # Skip if all values are NaN or empty
        if trsl.dropna().empty:
            continue

        # Standardization
        cml_min = trsl.min()
        cml_max = trsl.max()
        if pd.isna(cml_min) or pd.isna(cml_max) or cml_max - cml_min == 0:
            continue

        trsl_std = (trsl.values - cml_min) / (cml_max - cml_min)

        # Step detection
        conv = np.abs(np.convolve(trsl_std, step, mode="valid"))
        conv = np.append(np.append(np.zeros(500), conv), np.zeros(499))
        convDF = pd.DataFrame(conv, columns=["conv"])

        step_mask = conv > conv_threshold
        step_loc, _ = find_peaks(convDF.conv.where(step_mask), prominence=1)

        around_step = np.array([step_loc + (a - 5) for a in range(10)]).ravel()
        trsl.iloc[around_step] = np.nan

        step_loc = np.append(0, step_loc)
        for i in range(len(step_loc)):
            if i < len(step_loc) - 1:
                mean_val = trsl.iloc[step_loc[i] : step_loc[i + 1]].mean()
                trsl.iloc[step_loc[i] : step_loc[i + 1]] -= mean_val
            else:
                mean_val = trsl.iloc[step_loc[i] :].mean()
                trsl.iloc[step_loc[i] :] -= mean_val

        cml[col_name] = trsl

    return cml


def cml_reset_detect(cml: pd.DataFrame):
    """
    Detect cml reset by detecting stepdown in Uptime data. delete and interpolate values around

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.

    Returns
    cml : Pandas.DataFrame
    """

    # Get the indices where the uptime stepdown happens
    stepdown_mask = (cml["uptime_A"].diff() <= 0) | (cml["uptime_B"].diff() <= 0)
    stepdown_indices = cml.index[stepdown_mask]

    # (values around step)
    cml["trsl_A"][stepdown_indices] = np.nan
    cml["trsl_B"][stepdown_indices] = np.nan

    # interpolation both trsl, and R
    # cml = cml.interpolate(axis=0, method="linear", limit=20)
    cml = cml.dropna(axis=0, how="all", subset=["trsl_A", "trsl_B"])
    cml = cml.reset_index(drop=True)

    return cml


def subtract_trsl_median(cml: pd.DataFrame):
    """
    Subtract rolling median with window of 10000 samples from trsl data.
    This action supresses long therm variations and step changes

    Parameters
    cml: Pandas.DataFrame, containing cml data and reference WD

    Returns
    cml: Pandas.DataFrame, containing cml data and reference WD
    """
    cml["med_A"] = cml.trsl_A.rolling(window=10000, center=True).median()
    cml["med_B"] = cml.trsl_B.rolling(window=10000, center=True).median()

    # cml = cml.interpolate(axis=0, method="linear", limit_direction="both")

    cml["trsl_A"] = cml.trsl_A - cml.med_A
    cml["trsl_B"] = cml.trsl_B - cml.med_B
    cml = cml.drop(columns=["med_A", "med_B"])

    return cml


def ref_preprocess(
    cml: pd.DataFrame,
    sample_size=60,
    comp_lin_interp=False,
    upsampled_n_times=int(0),
    supress_single_zeros=False,
):
    """
    Create wet/dry flag from rain rate reference included in cml DataFrame.
    Optional features: Supress single zeros during light precipitation.
    Zero out last N-2 values of rainy periods, caused by upsampling
    and interpolation between last nonzero and first zero sample, vausing non zero leakage
    into zero values after rainy period.

    Parameters
    cml : Pandas.DataFrame containing cml data and corresponding reference raifall data
    sample_size : int, default = 30, downsampling coefficient for WD reference
    supress_single_zeros : bool, default = False, Supress single zeros if True
    comp_lin_interp : bool, default = False, Compensate for nonzero leakage if True
    upsampled_n_times : int, default value = 0, number of times upsampled. Defining
        number of samples to be zeroed.

    Returns
    cml : Pandas.DataFrame
    """
    # Supress single zeros during light precipitation
    if supress_single_zeros:
        nonzero_mask1 = cml.rain != 0
        shifted_mask_L = np.roll(nonzero_mask1, -1)
        shifted_mask_L[-1] = False  # Prevent wraparound issues
        shifted_mask_LL = np.roll(shifted_mask_L, -1)
        shifted_mask_LL[-1] = False  # Prevent wraparound issues
        shifted_mask_R = np.roll(nonzero_mask1, 1)
        shifted_mask_R[1] = False  # Prevent wraparound issues
        shifted_mask_RR = np.roll(shifted_mask_R, 1)
        shifted_mask_RR[1] = False  # Prevent wraparound issues

        single_zeros = np.where(
            (shifted_mask_L | shifted_mask_LL)
            & (shifted_mask_R | shifted_mask_RR)
            & ~nonzero_mask1
        )[0]

        cml.rain[single_zeros] = 0.01

    # Compensating linear interpolation:
    # Find indices where values transition from nonzero to zero (end of rain patterns)
    if comp_lin_interp & (upsampled_n_times >= 2):
        nonzero_mask2 = cml.rain != 0
        shifted_mask = np.roll(nonzero_mask2, -1)
        shifted_mask[-1] = False  # Prevent wraparound issues

        last_indices = np.where(nonzero_mask2 & ~shifted_mask)[0]

        # Zero out interpolated nonzero values at the end of rain patter
        for idx in last_indices:
            cml.rain[max(0, idx - (upsampled_n_times - 2)) : idx + 1] = (
                0  # Ensure we don't go out of bounds
            )

    # get downsampled R values, but keep them repeated with 30s
    cml["time"] = pd.to_datetime(cml["time"])
    rain_rate = (
        cml.resample(str(sample_size / 2) + "min", on="time").sum() / 20
    )  # /20 to convert [mm/10min] to [mm/30min]
    rain_rate = np.repeat(rain_rate.rain.values, sample_size)
    cml["rain"] = rain_rate[: (len(cml) - len(rain_rate))]

    # create reference WD flag
    cml["ref_wd"] = cml.rain > 0.15
    cml["ref_wd"][0] = False

    return cml


def balance_wd_classes(cml: pd.DataFrame, max_zero_length=600):
    """
    Exclude large dry periods from both rainfall and cml data to balance wet and dry classes
    built with a help of chatgpt: https://chatgpt.com/c/67daedee-2a1c-800a-9e0a-a82adf19d44b

    Parameters
    cml: Pandas.DataFrame, containing cml data and reference WD
    max_zero_length :int, default= 600, max length of 0 sequences to keep

    Returns
    cml_balanced: Pandas.DataFrame, containing cml data and reference WD
    """
    # Keep number of zeros around long zero segments
    keep_n_zeros = max_zero_length // 2

    # Mark contiguous segments of 0s and 1s
    cml["segment"] = (cml["ref_wd"].diff().ne(0)).cumsum()
    # get segment lengths
    segment_lengths = cml.groupby("segment")["ref_wd"].transform("count")

    # Find long zero segments
    long_zero_segments = cml[
        (cml["ref_wd"] == 0) & (segment_lengths > max_zero_length)
    ]["segment"].unique()

    # Create a mask to mark values to keep
    keep_mask = np.ones(
        len(cml), dtype=bool
    )  # mask of ones, excluded segments wil be marked 0

    for seg in long_zero_segments:
        seg_indices = cml[
            cml["segment"] == seg
        ].index  # Get all row indices of this segment

        # Keep first and last `buffer_size` zeros
        keep_indices = set(seg_indices[:keep_n_zeros]) | set(
            seg_indices[-keep_n_zeros:]
        )

        # Update mask
        middle_indices = set(seg_indices) - keep_indices
        keep_mask[list(middle_indices)] = False  # Exclude only the middle part

    # Apply the filter
    cml_balanced = cml[keep_mask].drop(columns=["segment"]).reset_index(drop=True)

    return cml_balanced


def cml_temp_extremes_std(cml: pd.DataFrame):
    """
    Remove fault extreme values in cml onchip temperature series by calculating floating window std,
    interpolate missing values

    Parameters
    cml : Pandas.DataFrame, containing two aligned adjacent CMLs and rainrate reference with timestamps.

    Returns
    cml : Pandas.DataFrame
    """

    # calculate rolling STD
    for temp in ["temperature_rx", "temperature_tx"]:
        temp_std = cml[temp].rolling(window=10, center=True).std()
        # Fill NaN values at the edges
        temp_std.fillna(method="bfill", inplace=True)
        temp_std.fillna(method="ffill", inplace=True)

        # drop values with STD above the threshold
        cml[temp] = cml[temp].where(np.abs(temp_std) < 2.0)

    # cml = cml.interpolate(axis=0, method="linear", limit_direction="both")
    return cml


def shuffle_dataset(cml: pd.DataFrame, segment_size=20000):
    """
    Separate cml dataframe into segments and shuffle them to supress long term dependent training,
    and shuffle training and testing data.
    built with a help of chatgpt: https://chatgpt.com/share/67f3ff55-b438-800a-8924-36e78669befe

    Parameters
    cml : Pandas.DataFrame, containing cml data with timestamps and reference WD
    segment_size : int, default = 20000, size of the segments to be shuffled,
        in which the data remain intact.

    Returns
    cml_shuffled : Pandas.DataFrame, containing cml data with timestamps and reference WD
    """

    cml["segment_id"] = cml.index // segment_size

    # Shuffle the segment IDs
    shuffled_segment_ids = np.random.permutation(cml["segment_id"].unique())

    # Map original segment IDs to shuffled IDs in a dictionary
    shuffle_map = {orig: new for new, orig in enumerate(shuffled_segment_ids)}

    # Shuffle the data by shuffled_segment_id (preserving intra-segment order)
    cml["shuffled_segment_id"] = cml["segment_id"].map(shuffle_map)
    cml_shuffled = cml.sort_values(["shuffled_segment_id", "time"]).reset_index(
        drop=True
    )

    # to sort cml back:
    # cml = cml.sort_values(['segment_id', 'time']).reset_index(drop=True)

    return cml_shuffled
