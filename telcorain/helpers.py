import numpy as np
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt
from functools import wraps
from numba import njit, prange
from shapely.geometry import Point as GeoPoint
from shapely.geometry.base import BaseGeometry
from PIL import Image
import configparser

from telcorain.handlers import logger


# Threshold boundaries
RAIN_THRESH = np.array(
    [
        0.1,
        0.115307,
        0.205048,
        0.364633,
        0.648420,
        1.153072,
        2.050483,
        3.646332,
        6.484198,
        11.53072,
        20.50483,
        36.46332,
        64.84198,
        115.3072,
    ]
)

# Colors matching each interval
RAIN_COLORS = np.array(
    [
        (57, 0, 112, 255),
        (47, 1, 169, 255),
        (0, 0, 252, 255),
        (0, 108, 192, 255),
        (0, 160, 0, 255),
        (0, 188, 0, 255),
        (52, 216, 0, 255),
        (156, 220, 0, 255),
        (224, 220, 0, 255),
        (252, 176, 0, 255),
        (252, 132, 0, 255),
        (252, 88, 0, 255),
        (252, 0, 0, 255),
        (160, 0, 0, 255),
    ],
    dtype=np.uint8,
)

TRANSPARENT = np.array([0, 0, 0, 0], dtype=np.uint8)


class MwLink:
    def __init__(
        self,
        link_id: int,
        name: str,
        tech: str,
        name_a: str,
        name_b: str,
        freq_a: int,
        freq_b: int,
        polarization: str,
        ip_a: str,
        ip_b: str,
        distance: float,
        latitude_a: float,
        longitude_a: float,
        latitude_b: float,
        longitude_b: float,
        dummy_latitude_a: float,
        dummy_longitude_a: float,
        dummy_latitude_b: float,
        dummy_longitude_b: float,
    ):
        self.link_id = link_id
        self.name = name
        self.tech = tech
        self.name_a = name_a
        self.name_b = name_b
        self.freq_a = freq_a
        self.freq_b = freq_b

        # since we can't handle cross polarization yet, let's consider them to have vertical polarization temporarilys
        if polarization == "X":
            polarization = "V"

        self.polarization = polarization
        self.ip_a = ip_a
        self.ip_b = ip_b
        self.distance = distance
        self.latitude_a = latitude_a
        self.longitude_a = longitude_a
        self.latitude_b = latitude_b
        self.longitude_b = longitude_b
        self.dummy_latitude_a = dummy_latitude_a
        self.dummy_longitude_a = dummy_longitude_a
        self.dummy_latitude_b = dummy_latitude_b
        self.dummy_longitude_b = dummy_longitude_b


@njit(parallel=True, fastmath=True)
def bbox_mask_numba(xgrid, ygrid, minx, miny, maxx, maxy):
    rows, cols = xgrid.shape
    mask = np.ones((rows, cols), dtype=np.bool_)
    for i in prange(rows):
        for j in range(cols):
            lon = xgrid[i, j]
            lat = ygrid[i, j]
            if lon < minx or lon > maxx or lat < miny or lat > maxy:
                mask[i, j] = False
    return mask


def mask_grid_fast_numba(data_grid, x_grid, y_grid, prep_poly, bbox):
    """
    Accelerated hybrid masking:
    - Numba performs very fast bounding-box masking
    - Shapely prepped polygon handles only filtered points
    """

    minx, miny, maxx, maxy = bbox

    # 1) Numba: extremely fast bbox mask
    bb_mask = bbox_mask_numba(x_grid, y_grid, minx, miny, maxx, maxy)
    # 2) Copy data
    out = data_grid.copy()
    # 3) Only check polygon for bbox-visible pixels
    rows, cols = x_grid.shape
    for i in range(rows):
        for j in range(cols):
            # If bbox already rejected → no Shapely call needed
            if not bb_mask[i, j]:
                out[i, j] = np.nan
                continue
            lon = x_grid[i, j]
            lat = y_grid[i, j]
            # Shapely prepared polygon → fast GEOS contains
            if not prep_poly.contains(GeoPoint(lon, lat)):
                out[i, j] = np.nan

    return out


def mask_grid_fast(data_grid, x_grid, y_grid, prep_poly, bbox):
    minx, miny, maxx, maxy = bbox
    out = data_grid.copy()

    rows, cols = x_grid.shape
    for i in range(rows):
        for j in range(cols):
            lon = x_grid[i, j]
            lat = y_grid[i, j]
            # bbox reject
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                out[i, j] = np.nan
                continue
            # polygon contains check
            if not prep_poly.contains(GeoPoint(lon, lat)):
                out[i, j] = np.nan

    return out


def mask_grid(
    data_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    polygons: list[BaseGeometry],
) -> np.ndarray:
    """
    Mask the 2D data grid with polygons. If a point is not within any of the polygons, it is set to NaN.

    :param data_grid: 2D ndarray data grid to be masked
    :param x_grid: 2D ndarray of x coordinates
    :param y_grid: 2D ndarray of y coordinates
    :param polygons: list of shapely Polygon (or similar) geometries
    :return: masked 2D ndarray with NaN values outside the polygons
    """
    mask = np.vectorize(
        lambda lon, lat: any(poly.contains(GeoPoint(lon, lat)) for poly in polygons)
    )
    within_polygons = mask(x_grid, y_grid)
    data_grid = data_grid.copy()
    data_grid[~within_polygons] = np.nan
    return data_grid


def rain_to_rgba(grid: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of a rainfall (mm/h) grid to RGBA image.
    """

    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)

    # mask NaNs and very low rain
    mask = (~np.isnan(grid)) & (grid >= 0.1)

    # assign bin index for each valid pixel
    bins = np.digitize(grid[mask], RAIN_THRESH)

    # map bin index → RGBA color
    rgba[mask] = RAIN_COLORS[bins]

    return rgba


def rain_to_rgba_exact(grid: np.ndarray) -> np.ndarray:
    rgba = np.zeros(grid.shape + (4,), dtype=np.uint8)
    # transparent mask
    mask = (~np.isnan(grid)) & (grid >= 0.1)
    # digitize using right=True to match "<" right boundary behavior
    bins = np.digitize(grid[mask], RAIN_THRESH, right=False)
    bins = bins - 1
    # anything < 0.1 or < first threshold stays transparent
    rgba[mask] = RAIN_COLORS[bins]

    return rgba


def ndarray_to_png(array: np.ndarray, output_path: str):
    """
    Convert 2D numpy array into a PNG image using fast vectorized rain-color mapping.
    """
    # rgba = rain_to_rgba(array)
    rgba = rain_to_rgba_exact(array)

    # vertical flip (same as your original implementation)
    rgba = np.flipud(rgba)
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_path, "PNG")


# def _get_rain_color(value: float) -> tuple[int, int, int, int]:
#     """
#     Get RGBA color tuple based on the rain intensity value.
#     The color scale is identical to the CHMI rain scale colorbar. The rain intensity values are in mm/h, the scale is
#     derived from the CHMI radar scale, where dBZ have been converted to mm/h using the Marshall-Palmer formula:
#     https://rdrr.io/github/potterzot/kgRainPredictR/man/marshall_palmer.html

#     [dBZ]  [mm/h]    [RGBA]
#     4      0.064842  (57, 0, 112, 255)
#     8      0.115307  (47, 1, 169, 255)
#     12     0.205048  (0, 0, 252, 255)
#     16     0.364633  (0, 108, 192, 255)
#     20     0.648420  (0, 160, 0, 255)
#     24     1.153072  (0, 188, 0, 255)
#     28     2.050483  (52, 216, 0, 255)
#     32     3.646332  (156, 220, 0, 255)
#     36     6.484198  (224, 220, 0, 255)
#     40     11.53072  (252, 176, 0, 255)
#     44     20.50483  (252, 132, 0, 255)
#     48     36.46332  (252, 88, 0, 255)
#     52     64.84198  (252, 0, 0, 255)
#     56     115.3072  (160, 0, 0, 255)

#     :param value: rain intensity value
#     :return: RGBA color tuple, NaN and values below 0.1 are transparent
#     """
#     if np.isnan(value) or value < 0.1:
#         return 0, 0, 0, 0  # transparent
#     elif 0.1 <= value < 0.115307:
#         return 57, 0, 112, 255
#     elif 0.115307 <= value < 0.205048:
#         return 47, 1, 169, 255
#     elif 0.205048 <= value < 0.364633:
#         return 0, 0, 252, 255
#     elif 0.364633 <= value < 0.648420:
#         return 0, 108, 192, 255
#     elif 0.648420 <= value < 1.153072:
#         return 0, 160, 0, 255
#     elif 1.153072 <= value < 2.050483:
#         return 0, 188, 0, 255
#     elif 2.050483 <= value < 3.646332:
#         return 52, 216, 0, 255
#     elif 3.646332 <= value < 6.484198:
#         return 156, 220, 0, 255
#     elif 6.484198 <= value < 11.53072:
#         return 224, 220, 0, 255
#     elif 11.53072 <= value < 20.50483:
#         return 252, 176, 0, 255
#     elif 20.50483 <= value < 36.46332:
#         return 252, 132, 0, 255
#     elif 36.46332 <= value < 64.84198:
#         return 252, 88, 0, 255
#     elif 64.84198 <= value < 115.3072:
#         return 252, 0, 0, 255
#     elif value >= 115.3072:
#         return 160, 0, 0, 255
#     else:
#         return 0, 0, 0, 0  # default: transparent


# def ndarray_to_png(array: np.ndarray, output_path: str):
#     """
#     Convert 2D numpy array into a PNG image.

#     :param array: 2D numpy array with rain intensity values
#     :param output_path: path to the output PNG image
#     """
#     height, width = array.shape
#     image = Image.new("RGBA", (width, height))
#     pixels = image.load()

#     # assign colors to pixels based on the values in the ndarray
#     for i in range(width):
#         for j in range(height):
#             # save pixel at height-j-1 Y position due to vertical flip
#             pixels[i, height - j - 1] = _get_rain_color(array[j, i])

#     try:
#         image.save(output_path, "PNG")
#     except Exception as error:
#         logger.error("Cannot save PNG image '%s': %s", output_path, error)


def save_ndarray_to_file(array: np.ndarray, output_path: str):
    """
    Save a 2D numpy ndarray to a local file.

    :param array: 2D numpy array to be saved
    :param output_path: Path to the file where the array will be saved
    """
    try:
        np.save(output_path, array)
    except Exception as error:
        logger.error('Cannot save ndarray to file "%s": %s', output_path, error)


def read_from_ndarray_file(input_path: str) -> np.ndarray:
    """
    Read a 2D numpy ndarray from a local file.

    :param input_path: Path to the saved ndarray file
    :return: The 2D numpy ndarray read from the file.
    """
    try:
        array: np.ndarray = np.round(np.load(input_path), decimals=3)
    except FileNotFoundError as error:
        logger.error(
            'Cannot read stored ndarray file "%s": File not found.', input_path
        )
        raise error
    except Exception as error:
        logger.error('Cannot read stored ndarray file "%s": %s', input_path, error)
        raise error

    return array


def read_value_from_ndarray_file(
    input_path: str,
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    total_rows: int,
    total_cols: int,
) -> np.number:
    """
    Read a value from a saved 2D numpy ndarray local file based on given geographic coordinates.

    :param input_path: Path to the saved ndarray file
    :param x: Longitude value to read
    :param y: Latitude value to read
    :param x_min: Minimum longitude (array border)
    :param x_max: Maximum longitude (array border)
    :param y_min: Minimum latitude (array border)
    :param y_max: Maximum latitude (array border)
    :param total_rows: Total number of rows in the array (vertical resolution)
    :param total_cols: Total number of columns in the array (horizontal resolution)
    :return: The value at the specified geographic coordinates in the array.
    """
    try:
        array: np.ndarray = np.load(input_path)
    except FileNotFoundError as error:
        logger.error(
            'Cannot read stored ndarray file "%s": File not found.', input_path
        )
        raise error
    except Exception as error:
        logger.error('Cannot read stored ndarray file "%s": %s', input_path, error)
        raise error

    x_step = (x_max - x_min) / (total_cols - 1)
    y_step = (y_max - y_min) / (total_rows - 1)

    # calculate the closest row and column indices
    col = round((x - x_min) / x_step)
    row = round((y - y_min) / y_step)

    # ensure indices are within array bounds
    col = min(max(col, 0), total_cols - 1)
    row = min(max(row, 0), total_rows - 1)

    return array[row, col]


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


def to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # treat naive input as UTC
        return dt.replace(tzinfo=None)
    # convert aware → UTC, then drop tzinfo
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


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


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # treat naive as UTC
        return dt.replace(tzinfo=timezone.utc)
    # convert any other tz to UTC
    return dt.astimezone(timezone.utc)


def create_config_dict(path: str, format: bool = True) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)

    cfg = {}
    for section in parser.sections():
        cfg[section] = {key: cast_value(value) for key, value in parser.items(section)}

    if format:
        cfg["time"]["start"] = datetime.fromisoformat(cfg["time"]["start"]).replace(
            tzinfo=timezone.utc
        )
        cfg["time"]["end"] = datetime.fromisoformat(cfg["time"]["end"]).replace(
            tzinfo=timezone.utc
        )

    return cfg


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


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now(tz=timezone.utc)
        result = func(*args, **kwargs)
        logger.debug(
            f"Function '{func.__name__}' executed in {datetime.now(tz=timezone.utc) - start_time}"
        )
        return result

    return wrapper
