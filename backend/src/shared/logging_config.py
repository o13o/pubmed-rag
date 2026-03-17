"""Structured JSON logging configuration.

Call setup_logging() once at application startup (API or CLI).
"""

import logging
import logging.config

from pythonjsonlogger.json import JsonFormatter


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON formatter to stderr."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
                "fmt": "%(timestamp)s %(levelname)s %(name)s %(message)s",
                "rename_fields": {"levelname": "levelname", "name": "name"},
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        "loggers": {
            "httpcore": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "pymilvus": {"level": "WARNING"},
            "litellm": {"level": "WARNING"},
            "openai": {"level": "WARNING"},
            "sentence_transformers": {"level": "WARNING"},
        },
    }
    logging.config.dictConfig(config)
