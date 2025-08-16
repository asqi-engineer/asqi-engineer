import logging
import os
from typing import Optional

asqi_logger = logging.getLogger("asqi")


def configure_logging(
    app_log_level: Optional[str] = None,
    lib_log_level: str = "WARNING",
    integrate_dbos: bool = True,
) -> None:
    """
    Configure centralized logging for the entire application.

    This is the single entry point for all logging setup. It configures a
    root handler for unified output and allows setting different log levels
    for the application's code versus third-party libraries.

    Args:
        app_log_level: Log level for the 'asqi' namespace (DEBUG, INFO, etc.).
                       Defaults to ASQI_LOG_LEVEL env var, or "INFO".
        lib_log_level: Log level for all other loggers (e.g., third-party libraries).
        integrate_dbos: If True, attempt to integrate with DBOS observability.
    """
    # Determine the application's log level
    if app_log_level is None:
        app_log_level = os.environ.get("ASQI_LOG_LEVEL", "INFO").upper()

    # Clear existing handlers to ensure idempotency
    root_logger = logging.getLogger()
    root_logger.setLevel(lib_log_level.upper())
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a single, formatted console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] (%(name)s:%(filename)s:%(lineno)s) %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    asqi_logger.setLevel(app_log_level.upper())

    asqi_logger.debug(
        f"Root logging configured. Library level: {lib_log_level}, App level: {app_log_level}."
    )

    if integrate_dbos:
        try:
            from dbos._logger import (
                add_otlp_to_all_loggers,
                add_transformer_to_all_loggers,
            )

            add_transformer_to_all_loggers()
            add_otlp_to_all_loggers()
            asqi_logger.debug("DBOS logging integration setup completed.")

        except ImportError:
            asqi_logger.warning("DBOS not found. Continuing without trace correlation.")
        except Exception as e:
            asqi_logger.warning(f"Failed to setup DBOS logging integration: {e}")
