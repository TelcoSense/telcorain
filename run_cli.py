import os
from warnings import simplefilter, filterwarnings

filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

from telcorain.custom_data import is_influx_source
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
        "4h": timedelta(hours=4),
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
        self.metadata_refresh_interval_runs: int = int(
            self.config["realtime"].get("metadata_refresh_interval_runs", 60)
        )

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
        if not is_influx_source(self.config):
            self.logger.error(
                "run_cli.py supports only [data_source] mode=influx. Use run_custom.py for pycomlink/custom datasets."
            )
            return

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
            exclude_cmls_path=self.config["cml"].get("exclude_cmls_path"),
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

        links = self._load_realtime_metadata()
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
        writer = Writer(
            influx_man=self.influx_man,
            write_influx_intensities=self.config["setting"]["write_influx_intensities"],
            config=self.config,
            since_time=datetime.min.replace(tzinfo=timezone.utc),
            is_historic=False,
        )

        realtime_window = self.config["realtime"]["realtime_timewindow"]

        while True:
            self._maybe_refresh_realtime_metadata(calculation)
            self._run_iteration(
                calculation,
                realtime_timewindow=realtime_window,
                writer=writer,
            )

    def _run_iteration(
        self,
        calculation: Calculation,
        realtime_timewindow: str,
        writer: Writer | None = None,
    ) -> None:
        """Run a single realtime iteration."""
        current_time, next_time, since_time = self._get_times()

        # Cleanup old data (always counts existing files; only deletes if enabled)
        removed_files, kept_files = self._cleanup_old_files(
            current_time=current_time,
            clean_raw=self.config["directories"]["clean_raw"],
            clean_web=self.config["directories"]["clean_web"],
        )
        if removed_files != 0:
            self.logger.info(
                "Cleanup: removed %d files, kept %d files", removed_files, kept_files
            )

        # Fetch data and run calculation
        calculation.run(realtime_timewindow=realtime_timewindow)

        # Writer
        if writer is None:
            writer = Writer(
                influx_man=self.influx_man,
                write_influx_intensities=self.config["setting"][
                    "write_influx_intensities"
                ],
                config=self.config,
                since_time=since_time,
                is_historic=False,
            )
        else:
            writer.set_since_time(since_time)

        writer.push_results(
            rain_grids=calculation.rain_grids,
            x_grid=calculation.x_grid,
            y_grid=calculation.y_grid,
            calc_dataset=calculation.calc_data_steps,
            rain_grids_sum=calculation.rain_grids_sum,
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
            f"Interpolation: res {self.config['interp']['interp_res']}",
            f"Using mercator: {self.config['interp']['use_mercator']}",
            f"Grid if used: {self.config['interp']['grid_step_m']}",
            f"power {self.config['interp']['idw_power']}",
            f"Realtime window: {self.config['realtime']['realtime_timewindow']}",
            f"Retention window: {self.config['realtime']['retention_window']}",
            f"Metadata refresh runs: {self.metadata_refresh_interval_runs}",
            f"CML quality filter: {self.config.get('cml_filter', {}).get('enabled', False)}",
            f"CML quality filter runs: {self.config.get('cml_filter', {}).get('realtime_interval_runs', 1000)}",
        ]

        logger.debug("Global config settings: " + "; ".join(config_info))
        logger.debug("Calculation settings: " + "; ".join(calc_info))

    def _get_times(self) -> tuple[datetime, datetime, datetime]:
        """Get current, next, and since times for calculation."""
        current_time = datetime.now(tz=timezone.utc)
        return (
            current_time,
            current_time + timedelta(seconds=self.repetition_interval),
            current_time - timedelta(seconds=self.repetition_interval),
        )

    def _load_realtime_metadata(self) -> dict:
        """Load the current realtime CML metadata selection from MariaDB."""
        return self.sql_man.load_metadata(
            min_length=self.config["cml"]["min_length"],
            max_length=self.config["cml"]["max_length"],
            exclude_ids=self.config["cml"]["exclude_cmls"],
            exclude_cmls_path=self.config["cml"].get("exclude_cmls_path"),
        )

    def _metadata_signature(self, links: dict) -> tuple:
        """
        Build a stable comparable snapshot of link metadata relevant for realtime processing.
        """
        return tuple(
            sorted(
                (
                    link_id,
                    link.ip_a,
                    link.ip_b,
                    link.freq_a,
                    link.freq_b,
                    link.polarization,
                    link.tech,
                    link.distance,
                    link.latitude_a,
                    link.longitude_a,
                    link.latitude_b,
                    link.longitude_b,
                )
                for link_id, link in links.items()
            )
        )

    def _maybe_refresh_realtime_metadata(self, calculation: Calculation) -> None:
        """
        Periodically reload MariaDB metadata so new/removed links can join without restart.
        """
        if self.metadata_refresh_interval_runs <= 0:
            return
        if calculation.realtime_runs <= 0:
            return
        if calculation.realtime_runs % self.metadata_refresh_interval_runs != 0:
            return

        current_signature = self._metadata_signature(calculation.links)
        refreshed_links = self._load_realtime_metadata()
        refreshed_signature = self._metadata_signature(refreshed_links)

        if refreshed_signature == current_signature:
            self.logger.info(
                "Metadata refresh after %d realtime runs: no link changes detected.",
                calculation.realtime_runs,
            )
            return

        current_ids = set(calculation.links)
        refreshed_ids = set(refreshed_links)
        added = len(refreshed_ids - current_ids)
        removed = len(current_ids - refreshed_ids)

        self.logger.info(
            "Metadata refresh after %d realtime runs: detected changes in the link set (+%d, -%d). "
            "Updating calculation metadata and forcing a full raw-data refresh.",
            calculation.realtime_runs,
            added,
            removed,
        )
        calculation.update_realtime_metadata(
            links=refreshed_links,
            selection=select_all_links(refreshed_links),
            reset_realtime_buffer=True,
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
                    stem = file_path.stem
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        stem = "_".join(
                            parts[:2]
                        )  # keep YYYY-MM-DD_HHMM, ignore suffix like _0.001
                    file_time = datetime.strptime(stem, "%Y-%m-%d_%H%M").replace(
                        tzinfo=timezone.utc
                    )
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
