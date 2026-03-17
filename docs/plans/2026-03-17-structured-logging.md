# Structured JSON Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `print()` statements in backend src with `logger` calls and configure JSON-formatted structured logging.

**Architecture:** Add a `logging_config.py` module that configures Python's `logging.dictConfig` with a JSON formatter via `python-json-logger`. Called once at API startup and at CLI entry point. Infrastructure scripts (`milvus_setup`, `export_collection`, `import_collection`) replace `print()` with `logger.info()`. CLI user-facing output uses `sys.stdout.write()`.

**Tech Stack:** python-json-logger, Python stdlib logging

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/shared/logging_config.py` | `setup_logging()` with JSON formatter via dictConfig |
| Create | `tests/unit/test_logging_config.py` | Unit tests for logging setup |
| Modify | `pyproject.toml` | Add `python-json-logger` dependency |
| Modify | `src/api/main.py:60-63` | Call `setup_logging()` in `create_app()` |
| Modify | `src/cli.py:42-45,89-110` | Replace `logging.basicConfig` with `setup_logging()`; `print()` → `sys.stdout.write()` |
| Modify | `src/ingestion/milvus_setup.py:99-105` | `print()` → `logger.info()` |
| Modify | `src/ingestion/export_collection.py:97-106` | `logging.basicConfig` → `setup_logging()`; `print()` → `logger.info()` |
| Modify | `src/ingestion/import_collection.py:65-75` | `logging.basicConfig` → `setup_logging()`; `print()` → `logger.info()` |

---

### Task 1: Add python-json-logger dependency

**Files:**
- Modify: `pyproject.toml:7-20`

- [ ] **Step 1: Add dependency**

Add `python-json-logger>=3.0` to the `dependencies` list in `pyproject.toml`.

- [ ] **Step 2: Install**

Run: `cd backend && uv sync`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add python-json-logger dependency"
```

---

### Task 2: Create logging_config.py with tests (TDD)

**Files:**
- Create: `src/shared/logging_config.py`
- Create: `tests/unit/test_logging_config.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_logging_config.py -v`
Expected: FAIL (ImportError — module does not exist)

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_logging_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/shared/logging_config.py tests/unit/test_logging_config.py
git commit -m "feat: add structured JSON logging configuration"
```

---

### Task 3: Wire setup_logging into API startup

**Files:**
- Modify: `src/api/main.py:1-2,60-63`

- [ ] **Step 1: Add import and call in create_app()**

In `src/api/main.py`, add `from src.shared.logging_config import setup_logging` to imports, then call `setup_logging()` as the first line inside `create_app()`.

- [ ] **Step 2: Run existing API tests to confirm no breakage**

Run: `cd backend && uv run pytest tests/unit/test_api_health.py tests/unit/test_api_ask.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat: wire JSON logging into API startup"
```

---

### Task 4: Replace print() in CLI with sys.stdout.write()

**Files:**
- Modify: `src/cli.py:42-45,89-110`

- [ ] **Step 1: Update cli.py**

Replace `logging.basicConfig(...)` with `setup_logging(level="DEBUG" if args.verbose else "INFO")`.

Replace all `print(...)` calls with `sys.stdout.write(... + "\n")`. Keep the exact same output format — this is user-facing terminal output, not log messages.

- [ ] **Step 2: Run full unit test suite to confirm no breakage**

Run: `cd backend && uv run pytest tests/unit/ -q`
Expected: 134 passed

- [ ] **Step 3: Commit**

```bash
git add src/cli.py
git commit -m "refactor: replace print() with sys.stdout.write() in CLI"
```

---

### Task 5: Replace print() in milvus_setup.py

**Files:**
- Modify: `src/ingestion/milvus_setup.py:99-105`

- [ ] **Step 1: Add module-level logger and replace print**

Add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file (after existing imports).

Replace line 105:
```python
print(f"Collection '{col.name}' ready. Fields: {[f.name for f in col.schema.fields]}")
```
with:
```python
logger.info("Collection '%s' ready. Fields: %s", col.name, [f.name for f in col.schema.fields])
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/unit/ -q`
Expected: 134 passed

- [ ] **Step 3: Commit**

```bash
git add src/ingestion/milvus_setup.py
git commit -m "refactor: replace print() with logger in milvus_setup"
```

---

### Task 6: Replace print() in export_collection.py and import_collection.py

**Files:**
- Modify: `src/ingestion/export_collection.py:97-106`
- Modify: `src/ingestion/import_collection.py:65-75`

- [ ] **Step 1: Update export_collection.py**

Replace `logging.basicConfig(...)` in `main()` with:
```python
from src.shared.logging_config import setup_logging
setup_logging()
```

Replace `print(f"Exported {total} records to {args.output}")` with:
```python
logger.info("Exported %d records to %s", total, args.output)
```

- [ ] **Step 2: Update import_collection.py**

Same pattern: replace `logging.basicConfig(...)` with `setup_logging()`, replace `print(...)` with `logger.info(...)`.

- [ ] **Step 3: Run tests**

Run: `cd backend && uv run pytest tests/unit/ -q`
Expected: 134 passed

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/export_collection.py src/ingestion/import_collection.py
git commit -m "refactor: replace print() with logger in ingestion scripts"
```

---

### Task 7: Final verification — zero print() in src/

- [ ] **Step 1: Grep for remaining print() calls**

Run: `grep -rn "print(" src/ --include="*.py"`
Expected: Zero matches

- [ ] **Step 2: Run full test suite**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: All 134+ tests pass

- [ ] **Step 3: Final commit (if any fixups needed)**
