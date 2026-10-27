"""Application logging configuration."""

from __future__ import annotations

import logging

from core.settings import get_settings


def configure_logging() -> None:
    """Configure standard logging once from application settings."""
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
