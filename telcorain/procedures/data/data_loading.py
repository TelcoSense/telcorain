import traceback
from os.path import exists
from datetime import datetime
from typing import Union
import numpy as np

from telcorain.database.influx_manager import InfluxManager
from telcorain.database.models.mwlink import MwLink
from telcorain.handlers.logging_handler import logger
from telcorain.procedures.exceptions import ProcessingException
from telcorain.procedures.utils.helpers import measure_time


@measure_time
def _get_ips_from_links_dict(selected_links: dict, links: dict) -> list[str]:
    if len(selected_links) < 1:
        raise ValueError("Empty selection array.")

    ips = []
    for link in selected_links:
        if link in links:
            # TODO: add dynamic exception list of constant Tx power devices
            # 1S10s have constant Tx power, so only one unit can be included in query
            # otherwise, both ends needs to be included in query, due Tx power correction
            if links[link].tech in ["1s10", "summit", "summit_bt"]:
                if selected_links[link] == 1:
                    ips.append(links[link].ip_a)
                elif selected_links[link] == 2:
                    ips.append(links[link].ip_b)
                elif selected_links[link] == 3:
                    ips.append(links[link].ip_a)
                    ips.append(links[link].ip_b)
            elif selected_links[link] == 0:
                continue
            else:
                ips.append(links[link].ip_a)
                ips.append(links[link].ip_b)

    return ips


@measure_time
def load_data_from_influxdb(
    influx_man: InfluxManager,
    cp: dict,
    selected_links: dict[int, int],
    links: dict[int, MwLink],
    log_run_id: str = "default",
    realtime: bool = False,
    force_data_refresh: bool = False,
) -> tuple[
    dict[str, Union[dict[str, dict[datetime, float]], str]], list[int], list[str]
]:
    try:
        ips = _get_ips_from_links_dict(selected_links, links)
        logger.info(
            "[%s] Querying InfluxDB for selected microwave links data...", log_run_id
        )

        # Realtime calculation is being done
        if realtime:
            logger.info("[%s] Realtime data procedure started.", log_run_id)
            influx_data = influx_man.query_units_realtime(
                ips,
                cp["realtime"]["realtime_timewindow"],
                cp["time"]["step"],
                cp["realtime"]["realtime_optimization"],
                force_data_refresh,
            )

        # In other case, notify we are doing historic calculation
        else:
            logger.info("[%s] Historic data procedure started.", log_run_id)
            influx_data = influx_man.query_units(
                ips,
                cp["time"]["start"],
                cp["time"]["end"],
                cp["time"]["step"],
            )

        diff = len(ips) - len(influx_data)
        logger.info(
            "[%s] Querying done. Got data of %d units, of total %d selected units.",
            log_run_id,
            len(influx_data),
            len(ips),
        )

        missing_links = []
        if diff > 0:
            logger.debug(
                "[%s] %d units are not available in selected time window:",
                log_run_id,
                diff,
            )
            for ip in ips:
                if ip not in influx_data:
                    for link in links:
                        if links[link].ip_a == ip:
                            logger.debug(
                                "[%s] Link: %d; Tech: %s; SIDE A: %s; IP: %s",
                                log_run_id,
                                links[link].link_id,
                                links[link].tech,
                                links[link].name_a,
                                links[link].ip_a,
                            )
                            missing_links.append(link)
                            break
                        elif links[link].ip_b == ip:
                            logger.debug(
                                "[%s] Link: %d; Tech: %s; SIDE B: %s; IP: %s",
                                log_run_id,
                                links[link].link_id,
                                links[link].tech,
                                links[link].name_b,
                                links[link].ip_b,
                            )
                            missing_links.append(link)
                            break
        return influx_data, missing_links, ips

    except BaseException as error:

        logger.error(
            "[%s] An unexpected error occurred during InfluxDB query: %s %s.\n"
            "Calculation thread terminated.",
            log_run_id,
            type(error),
            error,
        )

        traceback.print_exc()

        raise ProcessingException("Error occurred during InfluxDB query.")
