"""
Logging infrastructure:
- Rotating daily log files, 30-day retention
- Separate handlers for: trades, errors, model_retraining, edge_detections
- Structured log format
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LOG_DIR = Path(os.getenv("DATA_DIR", "data")) / "logs"


def setup_logging(log_level: str = "INFO"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console (force UTF-8 on Windows to avoid cp1252 encode errors) ───────
    if sys.platform == "win32":
        import io
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        console_h = logging.StreamHandler(utf8_stdout)
    else:
        console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(level)
    console_h.setFormatter(fmt)
    root.addHandler(console_h)

    # ── Main bot log (all levels) ─────────────────────────────────────────────
    _add_rotating_handler(root, "bot.log", logging.DEBUG, fmt)

    # ── Error log ─────────────────────────────────────────────────────────────
    err_h = _add_rotating_handler(root, "errors.log", logging.ERROR, fmt)

    # ── Specialized loggers ───────────────────────────────────────────────────
    _make_file_logger("trade_events", "trades.log", fmt)
    _make_file_logger("edge_detections", "edges.log", fmt)
    _make_file_logger("model_retraining", "model.log", fmt)


def _add_rotating_handler(
    logger: logging.Logger,
    filename: str,
    level: int,
    fmt: logging.Formatter,
) -> TimedRotatingFileHandler:
    h = TimedRotatingFileHandler(
        LOG_DIR / filename,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(fmt)
    logger.addHandler(h)
    return h


def _make_file_logger(name: str, filename: str, fmt: logging.Formatter) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    _add_rotating_handler(lg, filename, logging.DEBUG, fmt)
    return lg


# Convenience helpers
def log_trade_event(msg: str, **kwargs):
    lg = logging.getLogger("trade_events")
    if kwargs:
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        lg.info(f"{msg} | {extra}")
    else:
        lg.info(msg)


def log_edge(msg: str, **kwargs):
    lg = logging.getLogger("edge_detections")
    if kwargs:
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        lg.info(f"{msg} | {extra}")
    else:
        lg.info(msg)


def log_model(msg: str, **kwargs):
    lg = logging.getLogger("model_retraining")
    if kwargs:
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        lg.info(f"{msg} | {extra}")
    else:
        lg.info(msg)
