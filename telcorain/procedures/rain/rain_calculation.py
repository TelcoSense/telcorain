import traceback
from typing import Any

import numpy as np
from pycomlink.processing.baseline import baseline_constant
from pycomlink.processing.k_R_relation import calc_R_from_A
from pycomlink.processing.wet_antenna import (
    waa_schleiss_2013,
    waa_leijnse_2008_from_A_obs,
    waa_pastorek_2021_from_A_obs,
)
from xarray import Dataset

from telcorain.handlers import logger
from telcorain.procedures.exceptions import RaincalcException
from telcorain.procedures.rain import temperature_compensation
from telcorain.helpers import measure_time

from telcorain.procedures.wet_dry import cnn
from telcorain.procedures.wet_dry.cnn import CNN_OUTPUT_LEFT_NANS_LENGTH
from telcorain.procedures.wet_dry import preprocess_utility
from telcorain.procedures.wet_dry.cnn_utility import (
    cnn_infer_only,
    attach_cnn_output_to_xarray,
)


@measure_time
def get_rain_rates(
    calc_data: list[Dataset],
    config: dict[str, Any],
    ips: list[str],
    log_run_id: str = "default",
) -> list[Dataset]:
    """
    Compute rain rates for each link dataset in calc_data.

    - tsl, rsl, temperature_* are (channel_id, time)
    - frequency is a coordinate with dim (channel_id)
    - length and polarization are scalar coordinates
    """

    current_link = 0

    # switch for masking dry periods
    # If True  -> dry periods (wet == False/0) → NaN in R
    # If False -> dry periods remain as R computed from A (usually ~0 mm/h)
    ignore_dry_links = bool(config["wet_dry"].get("ignore_dry_links", False))

    try:
        logger.debug("[%s] Smoothing signal data...", log_run_id)
        count = 0

        # links stored in this list will be removed from the calculation
        # in case of enabled correlation filtering
        links_to_delete: list[Dataset] = []

        # ------------------------------------------------------------------
        # 1) Pre-processing: smoothing + temperature correlation / compensation
        # ------------------------------------------------------------------
        for link in calc_data:
            # Upper Tx power limit (40 dBm)
            link["tsl"] = link.tsl.astype(float).where(link.tsl < 40.0)
            link["tsl"] = link.tsl.astype(float).interpolate_na(
                dim="time", method="nearest", max_gap=None
            )

            # Bottom Rx power limit (-70 dBm) and removal of zeros
            link["rsl"] = (
                link.rsl.astype(float).where(link.rsl != 0.0).where(link.rsl > -70.0)
            )
            link["rsl"] = link.rsl.astype(float).interpolate_na(
                dim="time", method="nearest", max_gap=None
            )

            # Total attenuation (Tx – Rx)
            link["trsl"] = link.tsl - link.rsl

            # Temperatures – linear interpolation over time
            link["temperature_rx"] = link.temperature_rx.astype(float).interpolate_na(
                dim="time", method="linear", max_gap=None
            )
            link["temperature_tx"] = link.temperature_tx.astype(float).interpolate_na(
                dim="time", method="linear", max_gap=None
            )

            current_link += 1
            count += 1

            # --------------------------------------------------------------
            # Temperature-based filtering / compensation
            # --------------------------------------------------------------
            if config["temp"]["is_temp_filtered"]:
                logger.debug("[%s] Remove-link procedure started.", log_run_id)
                temperature_compensation.pearson_correlation(
                    count=count,
                    ips=ips,
                    curr_link=current_link,
                    link_todelete=links_to_delete,
                    link=link,
                    spin_correlation=config["temp"]["correlation_threshold"],
                )

            if config["temp"]["is_temp_compensated"]:
                logger.debug(
                    "[%s] Compensation algorithm procedure started.", log_run_id
                )
                temperature_compensation.compensation(
                    count=count,
                    ips=ips,
                    curr_link=current_link,
                    link=link,
                    spin_correlation=config["temp"]["correlation_threshold"],
                )
            current_link += 1

        # Remove links flagged for deletion (high temperature correlation)
        for link in links_to_delete:
            calc_data.remove(link)

        # ------------------------------------------------------------------
        # 2) Wet / Dry classification
        # ------------------------------------------------------------------
        logger.debug("[%s] Computing rain values...", log_run_id)
        current_link = 0

        if config["wet_dry"]["is_mlp_enabled"]:
            # CNN-based wet/dry classification
            if config["wet_dry"]["cnn_model"] == "polz":
                for link in calc_data:
                    # initialize wet flag
                    link["wet"] = (("time",), np.zeros([link.time.size]))

                    cnn_out = cnn.cnn_wet_dry(
                        trsl_channel_1=link.isel(channel_id=0).trsl.values,
                        trsl_channel_2=link.isel(channel_id=1).trsl.values,
                        threshold=0.82,
                        batch_size=128,
                    )

                    link["wet"] = (
                        ("time",),
                        np.where(np.isnan(cnn_out), link["wet"], cnn_out),
                    )

                # remove first CNN_OUTPUT_LEFT_NANS_LENGTH time values
                calc_data = [
                    link.isel(time=slice(CNN_OUTPUT_LEFT_NANS_LENGTH, None))
                    for link in calc_data
                ]
            else:
                for i, link in enumerate(calc_data):
                    preprocessed_df = preprocess_utility.cml_preprocess(
                        cml=link,
                        interp_max_gap=10,
                        suppress_step=True,
                        conv_threshold=250.0,
                        std_method=True,
                        window_size=10,
                        std_threshold=5.0,
                        z_method=True,
                        z_threshold=10.0,
                        reset_detect=False,
                        subtract_median=True,
                    )

                    cnn_out = cnn_infer_only(
                        preprocessed_df=preprocessed_df,
                        param_dir=config["wet_dry"]["cnn_model_name"],
                        sample_size=60,
                    )

                    link = attach_cnn_output_to_xarray(
                        link, cnn_out, sample_size=60, threshold=0.5
                    )
                    calc_data[i] = link

                logger.info(f"[{log_run_id}] Rain rates using custom CNN finished.")
        else:
            # Rolling-STD–based wet/dry detection (xarray rolling)
            for link in calc_data:
                link["wet"] = (
                    link.trsl.rolling(
                        time=config["wet_dry"]["rolling_values"],
                        center=config["wet_dry"]["is_window_centered"],
                    ).std(skipna=False)
                    > config["wet_dry"]["wet_dry_deviation"]
                )

        # ------------------------------------------------------------------
        # 3) Baseline, WAA, final attenuation and rain rate
        # ------------------------------------------------------------------
        for link in calc_data:
            # Ratio of wet periods
            link["wet_fraction"] = (link.wet == 1).sum() / len(link.time)

            # Baseline attenuation (constant)
            link["baseline"] = baseline_constant(
                trsl=link.trsl,
                wet=link.wet,
                n_average_last_dry=config["wet_dry"]["baseline_samples"],
            )

            # Rain-only attenuation
            link["A_rain"] = link.trsl - link.baseline
            link["A_rain"].values[link.A_rain < 0] = 0

            # Wet-antenna attenuation (WAA)
            if config["waa"]["waa_method"] == "schleiss":
                # NOTE: this matches the original delta_t expression
                delta_t = 60 / ((60 / config["time"]["step"]) * 60)
                link["waa"] = waa_schleiss_2013(
                    rsl=link.trsl,
                    baseline=link.baseline,
                    wet=link.wet,
                    waa_max=config["waa"]["waa_schleiss_val"],
                    delta_t=delta_t,
                    tau=config["waa"]["waa_schleiss_tau"],
                )
            elif config["waa"]["waa_method"] == "leijnse":
                link["waa"] = waa_leijnse_2008_from_A_obs(
                    A_obs=link.A_rain,
                    f_Hz=link.frequency * 1e9,
                    pol=link.polarization,
                    L_km=float(link.length),
                )
            elif config["waa"]["waa_method"] == "pastorek":
                link["waa"] = waa_pastorek_2021_from_A_obs(
                    A_obs=link.A_rain,
                    f_Hz=link.frequency * 1e9,
                    pol=link.polarization,
                    L_km=float(link.length),
                    A_max=2.2,
                )
            else:
                # fallback: clamp A (kept from original code)
                link["waa"] = link["A"]
                link["waa"] = link["waa"].where(link["waa"] >= 0, 0)

            # Final rain attenuation
            link["A"] = link.A_rain - link["waa"]
            link["A"] = link["A"].where(link["A"] >= 0, 0)

            # Rain intensity (mm/h)
            link["R"] = calc_R_from_A(
                A=link.A,
                L_km=float(link.length),
                f_GHz=link.frequency,  # DataArray
                pol=link.polarization,
            )

            # Optionally ignore dry periods by setting R to NaN
            if ignore_dry_links:
                wet = link["wet"]
                if wet.dtype == bool:
                    wet_mask = wet
                else:
                    wet_mask = wet > 0.5  # treat > 0.5 as wet
                link["R"] = link["R"].where(wet_mask, np.nan)

            current_link += 1

        return calc_data

    except BaseException as error:
        bad_link = None
        if 0 <= current_link < len(calc_data):
            bad_link = calc_data[current_link]

        logger.error(
            "[%s] An unexpected error occurred during rain calculation: %s %s.\n"
            "Last processed microwave link dataset:\n%s\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
            bad_link,
        )

        traceback.print_exc()
        raise RaincalcException("Error occurred during rainfall calculation processing")
