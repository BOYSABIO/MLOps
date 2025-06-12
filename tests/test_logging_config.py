"""Tests for the setup_logging function in logging_config module."""

import logging
from src.utils.logging_config import setup_logging


def test_logging_creates_log_file(tmp_path):
    """Check if log file is created and logging works."""
    log_path = tmp_path / "logs" / "main_log.log"

    setup_logging(log_file=str(log_path))

    logging.info("Test log entry")

    assert log_path.exists(), "Log file was not created"
    contents = log_path.read_text()
    assert "Test log entry" in contents


def test_logging_format_and_level(tmp_path, capsys):
    """Verify logging format and INFO level output to console."""
    log_path = tmp_path / "logs" / "main_log.log"
    setup_logging(log_file=str(log_path))

    logging.info("Check log format")
    captured = capsys.readouterr()
    assert "INFO" in captured.err
    assert "Check log format" in captured.err
