"""Logging configuration shared by the CLI and notebooks."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
QUIET_LOGGERS = ("httpx", "httpcore")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the process.

    HTTP client libraries are held at WARNING so per-request logs do not
    interrupt progress bars; retries and failures still surface.

    Args:
        level: Minimum level to emit, such as ``logging.INFO``.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))
