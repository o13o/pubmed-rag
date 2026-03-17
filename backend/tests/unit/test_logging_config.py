"""Tests for structured JSON logging configuration."""

import json
import logging

from src.shared.logging_config import setup_logging


def test_setup_logging_configures_json_format(capfd):
    """After setup_logging(), log output should be valid JSON."""
    setup_logging()
    logger = logging.getLogger("test.json_format")
    logger.info("hello structured logging")

    captured = capfd.readouterr()
    line = captured.err.strip().split("\n")[-1]
    data = json.loads(line)
    assert data["message"] == "hello structured logging"
    assert "timestamp" in data
    assert data["levelname"] == "INFO"


def test_setup_logging_includes_logger_name(capfd):
    setup_logging()
    logger = logging.getLogger("mymodule.sub")
    logger.warning("test warning")

    captured = capfd.readouterr()
    line = captured.err.strip().split("\n")[-1]
    data = json.loads(line)
    assert data["name"] == "mymodule.sub"
    assert data["levelname"] == "WARNING"


def test_setup_logging_debug_level(capfd):
    setup_logging(level="DEBUG")
    logger = logging.getLogger("test.debug")
    logger.debug("debug msg")

    captured = capfd.readouterr()
    line = captured.err.strip().split("\n")[-1]
    data = json.loads(line)
    assert data["message"] == "debug msg"


def test_setup_logging_suppresses_noisy_loggers(capfd):
    setup_logging()
    noisy = logging.getLogger("httpcore")
    noisy.debug("should be suppressed")

    captured = capfd.readouterr()
    assert "should be suppressed" not in captured.err
