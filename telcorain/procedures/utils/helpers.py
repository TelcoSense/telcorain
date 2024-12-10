import configparser
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt

import numpy as np

from telcorain.database.models.mwlink import MwLink


def calc_distance(lat_A: float, long_A: float, lat_B: float, long_B: float) -> float:
    """
    Calculate distance between two points on Earth.

    :param lat_A: latitude of point A in decimal degrees
    :param long_A: longitude of point A in decimal degrees
    :param lat_B: latitude of point B in decimal degrees
    :param long_B: longitude of point B in decimal degrees
    :return: distance in kilometers
    """
    # Approximate radius of earth in km
    r = 6373.0

    lat_A = radians(lat_A)
    long_A = radians(long_A)
    lat_B = radians(lat_B)
    long_B = radians(long_B)

    dlon = long_B - long_A
    dlat = lat_B - lat_A

    a = sin(dlat / 2) ** 2 + cos(lat_A) * cos(lat_B) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c


def dt64_to_unixtime(dt64: np.datetime64) -> int:
    """
    Convert numpy datetime64 to Unix timestamp.

    :param dt64: numpy datetime64
    :return: number of seconds since Unix epoch
    """
    unix_epoch = np.datetime64(0, "s")
    s = np.timedelta64(1, "s")
    return int((dt64 - unix_epoch) / s)


def utc_datetime(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        tzinfo=timezone.utc,
    )


def datetime_rfc(dt: datetime) -> str:
    """
    Convert datetime to string compliant with the RFC 3339

    :param dt: Python datetime object
    :return: RFC compliant datetime string
    """
    return dt.isoformat().replace("+00:00", "Z")


def cast_value(value):
    """
    Tries to cast the value to an appropriate type.
    Priority: int > float > bool > string
    """
    if value.lower() in ("true", "false"):  # Handle booleans
        return value.lower() == "true"
    try:
        return int(value)  # Try casting to int
    except ValueError:
        pass
    try:
        return float(value)  # Try casting to float
    except ValueError:
        pass
    return value  # Default to string if no other type matches


def create_cp_dict(path: str, format: bool = True) -> dict:
    config = configparser.ConfigParser()
    config.read(path)
    cp = {}        
    for section in config.sections():
        cp[section] = {key: cast_value(value) for key, value in config.items(section)}
    if format:
        cp["time"]["start"] = datetime.fromisoformat(cp["time"]["start"]).replace(
            tzinfo=timezone.utc
        )
        cp["time"]["end"] = datetime.fromisoformat(cp["time"]["end"]).replace(
            tzinfo=timezone.utc
        )
    return cp


def select_all_links(links: list[MwLink]) -> dict[int, int]:
    selected_links = {}
    for link in links:
        selected_links[links[link].link_id] = 3
    return selected_links


def select_links(link_ids: list[int]) -> dict[int, int]:
    selected_links = {}
    for link_id in link_ids:
        selected_links[link_id] = 3
    return selected_links
