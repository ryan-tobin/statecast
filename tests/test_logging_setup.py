import logging

from statecast.logging_setup import LOG_FORMAT, configure_logging


def test_configure_logging_sets_level_and_format() -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]

    configure_logging(logging.DEBUG)

    try:
        assert root_logger.level == logging.DEBUG
        formatter = root_logger.handlers[0].formatter
        assert formatter is not None
        assert formatter._fmt == LOG_FORMAT
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)


def test_configure_logging_quiets_http_client_logs() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    configure_logging(logging.INFO)

    try:
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)
