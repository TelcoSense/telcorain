import sys
import os
import argparse
from warnings import simplefilter, filterwarnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

from telcorain.database.influx_manager import influx_man
from telcorain.database.sql_manager import SqlManager
from telcorain.handlers import logger
from telcorain.writer import Writer
from telcorain.calculation import Calculation
from telcorain.helpers import create_config_dict, select_all_links

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=DeprecationWarning)
simplefilter(action="ignore", category=FutureWarning)


class TelcorainCLI:
    """
    Main class of TelcoRain CLI for raingrids computation.
    """

    delta_map: dict = {
        "1h": timedelta(hours=1),
        "3h": timedelta(hours=3),
        "6h": timedelta(hours=6),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "2d": timedelta(days=2),
        "7d": timedelta(days=7),
        "14d": timedelta(days=14),
        "30d": timedelta(days=30),
    }

    def __init__(self, config_path: str = "configs/config.ini") -> None:
        """Initialize CLI with configuration."""
        self.config: dict = create_config_dict(path=config_path, format=True)
        self.repetition_interval: int = self.config["setting"]["repetition_interval"]
        self.sleep_interval: int = self.config["setting"]["sleep_interval"]

        self.realtime_timewindow = self.delta_map[
            self.config["realtime"]["realtime_timewindow"]
        ].total_seconds()

        self.sql_man = SqlManager()
        self.influx_man = influx_man
        self.logger = logger

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def run(self, first: bool = False) -> None:
        """
        Run the TelcoRain calculation in continuous loop.

        If first is True, the first iteration uses retention_window instead of
        realtime_timewindow to save computation time.
        """
        if first:
            try:
                self._run_single_start(
                    realtime_window=self.config["realtime"]["retention_window"],
                    first_run_label="for first iteration on retention_window.",
                )
            except KeyboardInterrupt:
                logger.info("Shutdown of the program...")

        try:
            self._run_realtime_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown of the program...")

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _run_single_start(
        self, realtime_window: str, first_run_label: str = ""
    ) -> None:
        """Single startup iteration (for the `--first` mode)."""
        self._print_init_log_info()

        links = self.sql_man.load_metadata(
            min_length=self.config["cml"]["min_length"],
            exclude_ids=self.config["cml"]["exclude_cmls"],
        )
        selected_links = select_all_links(links=links)

        start_time = datetime.now(tz=timezone.utc)
        self.logger.info("Starting Telcorain CLI at %s %s", start_time, first_run_label)

        calculation = Calculation(
            influx_man=self.influx_man,
            links=links,
            selection=selected_links,
            config=self.config,
        )
        self._run_iteration(calculation, realtime_timewindow=realtime_window)

    def _run_realtime_loop(self) -> None:
        """Main infinite realtime loop."""
        self._print_init_log_info()

        links = self.sql_man.load_metadata(
            min_length=self.config["cml"]["min_length"],
            max_length=self.config["cml"]["max_length"],
            exclude_ids=self.config["cml"]["exclude_cmls"],
        )
        selected_links = select_all_links(links=links)

        start_time = datetime.now(tz=timezone.utc)
        self.logger.info("Starting Telcorain CLI at %s.", start_time)

        calculation = Calculation(
            influx_man=self.influx_man,
            links=links,
            selection=selected_links,
            config=self.config,
            is_historic=False,
        )

        realtime_window = self.config["realtime"]["realtime_timewindow"]

        while True:
            self._run_iteration(calculation, realtime_timewindow=realtime_window)

    def _run_iteration(
        self, calculation: Calculation, realtime_timewindow: str
    ) -> None:
        """Run a single realtime iteration."""
        current_time, next_time, since_time = self._get_times()

        # Cleanup old data (always counts existing files; only deletes if enabled)
        removed_files, kept_files = self._cleanup_old_files(
            current_time=current_time,
            clean_raw=self.config["directories"]["clean_raw"],
            clean_web=self.config["directories"]["clean_web"],
        )
        self.logger.info(
            "Cleanup: removed %d files, kept %d files", removed_files, kept_files
        )

        # Fetch data and run calculation
        calculation.run(realtime_timewindow=realtime_timewindow)

        # Writer
        writer = Writer(
            influx_man=self.influx_man,
            skip_influx=self.config["realtime"]["skip_influx_write"],
            config=self.config,
            since_time=since_time,
            is_historic=False,
        )

        writer.push_results(
            rain_grids=calculation.rain_grids,
            x_grid=calculation.x_grid,
            y_grid=calculation.y_grid,
            calc_dataset=calculation.calc_data_steps,
        )

        self.logger.info("RUN ends. Next iteration should start at: %s.", next_time)
        self.logger.info(
            "Final time of calculation: %s",
            datetime.now(tz=timezone.utc) - current_time,
        )
        self.logger.info("...sleeping until %s UTC time...", next_time)

        while datetime.now(tz=timezone.utc) < next_time:
            sleep(self.sleep_interval)

    def _print_init_log_info(self) -> None:
        """Log initial configuration information."""
        config_info = [
            f"Logger level: {self.config['logging']['init_level']}",
            f"MariaDB: {self.config['mariadb']['address']}:{self.config['mariadb']['port']}",
            f"InfluxDB: {self.config['influx2']['url']}",
            f"Output folders -- log: {self.config['directories']['logs']}",
            f"web: {self.config['directories']['outputs_web']}",
            f"raw: {self.config['directories']['outputs_raw']}",
        ]

        calc_info = [
            f"Step: {self.config['time']['step']}",
            f"IsMLPEnabled: {self.config['wet_dry']['is_mlp_enabled']}",
            f"WAA method: {self.config['waa']['waa_method']}",
            f"Interpolation: res {self.config['interp']['interp_res']}, "
            f"power {self.config['interp']['idw_power']}",
            f"Realtime window: {self.config['realtime']['realtime_timewindow']}",
            f"Retention window: {self.config['realtime']['retention_window']}",
        ]

        logger.info("Global config settings: " + "; ".join(config_info))
        logger.info("Calculation settings: " + "; ".join(calc_info))

    def _get_times(self) -> tuple[datetime, datetime, datetime]:
        """Get current, next, and since times for calculation."""
        current_time = datetime.now(tz=timezone.utc)
        return (
            current_time,
            current_time + timedelta(seconds=self.repetition_interval),
            current_time - timedelta(seconds=self.repetition_interval),
        )

    # ======================================================================
    # CLEANUP
    # ======================================================================

    def _cleanup_old_files(
        self,
        current_time: datetime,
        clean_raw: bool = True,
        clean_web: bool = True,
    ) -> tuple[int, int]:
        """
        Delete old files from raw and/or web output directories.

        Files are considered "old" if their timestamp (parsed from filename) is
        older than the retention threshold.

        Even if clean_raw / clean_web is False, files are still counted as "kept"
        so you see how many files are present.
        """
        retention_window = self.config["realtime"]["retention_window"]
        threshold = current_time - self.delta_map[retention_window]

        total_deleted = 0
        total_kept = 0

        def process_folder(
            folder: Path,
            do_cleanup: bool,
            label: str,
        ) -> tuple[int, int]:
            deleted = 0
            kept = 0

            if not folder.exists():
                return 0, 0

            for file_path in folder.glob("*"):
                if not file_path.is_file():
                    continue

                try:
                    file_time = datetime.strptime(
                        file_path.stem, "%Y-%m-%d_%H%M"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning(
                        "Cleanup[%s]: skipping non-timestamped file: %s",
                        label,
                        file_path.name,
                    )
                    kept += 1
                    continue

                # Decide delete/keep
                if do_cleanup and file_time < threshold:
                    try:
                        file_path.unlink()
                        deleted += 1
                    except OSError as e:
                        logger.error(
                            "Cleanup[%s]: failed to delete %s: %s",
                            label,
                            file_path,
                            e,
                        )
                        # if deletion fails, treat as kept
                        kept += 1
                else:
                    kept += 1

            logger.debug(
                "Cleanup[%s]: folder %s → deleted=%d, kept=%d",
                label,
                folder,
                deleted,
                kept,
            )
            return deleted, kept

        # Raw folder
        raw_dir = Path(self.config["directories"]["outputs_raw"])
        d_raw, k_raw = process_folder(raw_dir, clean_raw, "raw")
        total_deleted += d_raw
        total_kept += k_raw

        # Web folder
        web_dir = Path(self.config["directories"]["outputs_web"])
        d_web, k_web = process_folder(web_dir, clean_web, "web")
        total_deleted += d_web
        total_kept += k_web

        return total_deleted, total_kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "TelcoRain CLI. It computes raingrids from CML Influx data "
            "and saves results to a local folder (.npy and .png), "
            "optionally to MariaDB and InfluxDB."
        )
    )

    parser.add_argument(
        "--run",
        action="store_true",
        default=True,
        help="Run the CLI calculation.",
    )

    parser.add_argument(
        "--first",
        action="store_true",
        default=False,
        help="Run with the retention_window first and then with realtime_timewindow.",
    )

    args = parser.parse_args()
    telco_cli = TelcorainCLI()
    if args.first:
        telco_cli.run(first=True)
    if args.run:
        telco_cli.run()
