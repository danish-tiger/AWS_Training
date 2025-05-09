import logging
import sys
from typing import Optional


def configure_logging(
    log_level: str = "INFO",
    log_path: Optional[str] = None,
    console_log: bool = True,
):
    """Configure logging for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_path: Path to log file. If None, no file logging.
        console_log: Whether to log to console.
    """
    handlers = []

    if console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
    )
