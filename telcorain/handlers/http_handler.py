"""Module containing the HTTP server related functions."""

import json
import os
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

# from telcorain.database.sql_manager import sql_man
from telcorain.handlers import config_handler
from telcorain.handlers.logging_handler import logger
from telcorain.handlers.writer import (
    read_from_ndarray_file,
    read_value_from_ndarray_file,
)


def qs_parse_time_and_parameters(
    query_strings: dict[str, list[str]]
) -> tuple[datetime, int]:
    """
    Parses the timestamp and parameters from the query strings.

    :param query_strings: The query strings.
    :return: A tuple containing the timestamp and parameters.
    """
    timestamp = query_strings.get("timestamp", [None])[0]
    parameters = query_strings.get("parameters", [None])[0]

    if not timestamp or not parameters:
        raise ValueError(
            "Missing one or more required parameters, check: timestamp, parameters"
        )
    else:
        # for some less smart linters like the one in PyCharm, cast is needed
        timestamp = cast(str, timestamp)
        parameters = cast(str, parameters)

    try:
        p_timestamp = datetime.strptime(timestamp, "%Y-%m-%d_%H%M")
    except ValueError:
        raise ValueError("Invalid timestamp format, correct form: YYYY-MM-DD_HHMM")

    if not parameters.isdigit() or int(parameters) < 1:
        raise ValueError("Invalid parameters ID, must be positive integer")
    else:
        p_parameters = int(parameters)

    return p_timestamp, p_parameters


def qs_parse_coordinates(query_strings: dict[str, list[str]]) -> tuple[float, float]:
    """
    Parses the latitude and longitude from the query strings.

    :param query_strings: The query strings.
    :return: A tuple containing the latitude and longitude.
    """
    latitude = query_strings.get("latitude", [None])[0]
    longitude = query_strings.get("longitude", [None])[0]

    if not latitude or not longitude:
        raise ValueError(
            "Missing one or more required parameters, check: latitude, longitude"
        )
    else:
        # for some less smart linters like the one in PyCharm, cast is needed
        latitude = cast(str, latitude)
        longitude = cast(str, longitude)

    if (
        not latitude.replace(".", "", 1).isdigit()
        or not longitude.replace(".", "", 1).isdigit()
    ):
        raise ValueError("Invalid latitude or longitude, must be decimal float")
    else:
        p_latitude = float(latitude)
        p_longitude = float(longitude)
        if (
            p_latitude < -90
            or p_latitude > 90
            or p_longitude < -180
            or p_longitude > 180
        ):
            raise ValueError(
                "Invalid latitude or longitude, must be in range: latitude (-90, 90), longitude (-180, 180)"
            )

    return p_latitude, p_longitude


def handler_class(sql_man, outputs_dir, outputs_raw_dir):
    class CustomHandler(TelcorainHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                sql_man=sql_man,
                outputs_dir=outputs_dir,
                outputs_raw_dir=outputs_raw_dir,
                **kwargs,
            )

    return CustomHandler


class TelcorainHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the Telcorain application."""

    def __init__(
        self, *args, sql_man=None, outputs_dir=None, outputs_raw_dir=None, **kwargs
    ):
        self.sql_man = sql_man
        self.outputs_dir = outputs_dir
        self.outputs_raw_dir = outputs_raw_dir

        super().__init__(*args, **kwargs)
        print(f"SQL man: {self.outputs_dir}, CP: {self.outputs_raw_dir}")

    def __send_json_ok_response(self, response: dict[str, Any]):
        """
        Sends a JSON response with a 200 status code.

        :param response: The response to send.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def __send_json_error_response(self, code: int, message: str):
        """
        Sends a JSON response with an error status code.

        :param code: The HTTP status code.
        :param message: THe error message.
        """
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {"status": "error", "code": code, "error": message}
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self):
        """Overrides the do_GET method to handle the custom API endpoints."""
        parsed_url = urlparse(self.path)
        query_strings = parse_qs(parsed_url.query)

        if parsed_url.path == "/api/grid":
            try:
                timestamp, parameters = qs_parse_time_and_parameters(query_strings)

                # verify existence of the requested data
                if self.sql_man.verify_raingrid(parameters, timestamp):
                    # read the whole grid from the raw outputs directory
                    grid = read_from_ndarray_file(
                        input_path=f"{self.outputs_raw_dir}/"
                        f"{query_strings.get('timestamp', [None])[0]}.npy"
                    )

                    response = {
                        "grid": grid.tolist(),
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "parameters": parameters,
                    }
                    self.__send_json_ok_response(response)
                else:
                    raise FileNotFoundError(
                        "No data available for the requested parameters and timestamp"
                    )
            except ValueError as e:
                self.send_error(400, str(e), json_response=True)
            except FileNotFoundError as e:
                self.send_error(404, str(e), json_response=True)
            except Exception as e:
                logger.error(
                    "Unexpected error during processing of /api/grid request: %s", e
                )
                self.send_error(
                    500,
                    "Unexpected internal error, check Telcorain log",
                    json_response=True,
                )
        elif parsed_url.path == "/api/gridvalue":
            try:
                timestamp, parameters = qs_parse_time_and_parameters(query_strings)
                latitude, longitude = qs_parse_coordinates(query_strings)

                # verify existence of the requested data
                if self.sql_man.verify_raingrid(parameters, timestamp):
                    # get the given calculation parameters from the database
                    params = self.sql_man.get_realtime(parameters_id=parameters)
                    # read the coordinates value from the raw outputs directory
                    value = read_value_from_ndarray_file(
                        input_path=f"{self.outputs_raw_dir}/"
                        f"{query_strings.get('timestamp', [None])[0]}.npy",
                        x=longitude,
                        y=latitude,
                        x_min=params["X_MIN"],
                        x_max=params["X_MAX"],
                        y_min=params["Y_MIN"],
                        y_max=params["Y_MAX"],
                        total_cols=params["X_count"],
                        total_rows=params["Y_count"],
                    )

                    response = {
                        "value": round(value, 4),
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                        "parameters": parameters,
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                    self.__send_json_ok_response(response)
                else:
                    raise FileNotFoundError(
                        "No data available for the requested parameters and timestamp"
                    )
            except ValueError as e:
                self.send_error(400, str(e), json_response=True)
            except FileNotFoundError as e:
                self.send_error(404, str(e), json_response=True)
            except Exception as e:
                logger.error(
                    "Unexpected error during processing of /api/gridvalue request: %s",
                    e,
                )
                self.send_error(
                    500,
                    "Unexpected internal error, check Telcorain log",
                    json_response=True,
                )
        elif parsed_url.path == "/api/hello":
            response = {
                "status": "ok",
                "message": "It works! Telcorain is running and wating for your requests.",
            }
            self.__send_json_ok_response(response)
        else:
            super().do_GET()

    def log_message(self, format: str, *args):
        """Overrides the log_message method to log the client's IP address and requested path."""
        if self.path == "/":
            logger.debug(
                'HTTP is serving directory listing to client: "%s" on port: %d.',
                self.client_address[0],
                self.client_address[1],
            )
        elif self.path.startswith("/api/"):
            logger.debug(
                'HTTP is serving an API request for: "%s" to client: "%s" on port: %d.',
                self.path,
                self.client_address[0],
                self.client_address[1],
            )
        else:
            logger.debug(
                'HTTP is serving file: "%s%s" to client: "%s" on port: %d.',
                self.outputs_dir,
                self.path,
                self.client_address[0],
                self.client_address[1],
            )

    def log_error(self, format, *args):
        """Override the log_error method to do nothing since the error is already logged in send_error."""
        return

    def send_error(
        self,
        code: int,
        message: str = None,
        explain: str = None,
        json_response: bool = False,
    ):
        """Overrides the send_error method to log the error code and message together with client IP and file path."""
        logger.warning(
            'HTTP error %d during request for: "%s", client: "%s" on port: %d. %s.',
            code,
            self.path,
            self.client_address[0],
            self.client_address[1],
            message,
        )

        if json_response:
            self.__send_json_error_response(code, message)
        else:
            super().send_error(code, message, explain)


def setup_http_server(sql_man, config_db, outputs_dir, outputs_raw_dir):
    """
    Sets up and runs the HTTP server. Server has different endpoints:
    - serving image files from the web outputs directory
    - REST API endpoints utilizing functions reading data from the raw outputs directory
    Must be run in a separate thread.
    """
    is_enabled = config_db["http"]["enable_http_server"]
    address = config_db["http"]["http_server_address"]
    port = config_db["http"]["http_server_port"]

    if str(is_enabled).lower() == "true":
        logger.info("Starting HTTP server...")

        try:
            if not os.path.exists(outputs_dir):
                os.makedirs(outputs_dir)
                logger.debug(
                    "Created %s directory for output PNG files.",
                    outputs_dir,
                )

            if not os.path.exists(outputs_raw_dir):
                os.makedirs(outputs_raw_dir)
                logger.debug(
                    "Created %s directory for output raw files.",
                    outputs_raw_dir,
                )

            if address == "0.0.0.0":
                address_t = ""
            else:
                address_t = address
            socket = (address_t, port)

            # httpd = HTTPServer(socket, TelcorainHTTPRequestHandler)
            custom_handler = handler_class(sql_man, outputs_dir, outputs_raw_dir)
            httpd = HTTPServer(socket, custom_handler)
        except Exception as error:
            logger.error("Cannot start HTTP server due to an error: %s", error)
            return
        else:
            logger.debug(f"HTTP server is running on {address}:{port}.")
            logger.debug(f"HTTP server is serving files from directory: {outputs_dir}")
        # run the HTTP server
        httpd.serve_forever()
    else:
        logger.info("HTTP server is disabled.")


def start_http_server_thread(sql_man, config, config_db):
    """Starts the HTTP server in a separate thread."""
    outputs_dir = config["directories"]["outputs_web"]
    outputs_raw_dir = config["directories"]["outputs_raw"]
    http_server_thread = threading.Thread(
        target=setup_http_server,
        args=(sql_man, config_db, outputs_dir, outputs_raw_dir),
        daemon=True,
        name="HTTPServer",
    )
    http_server_thread.start()
    return http_server_thread
