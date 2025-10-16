#!/usr/bin/env python3
"""
Shared helpers for scripts in the work directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(debug: bool = False, name: str = "work") -> logging.Logger:
    """
    Create or retrieve a logger with a unified format for the work scripts.
    Subsequent calls reuse the handler to avoid duplicated log output.
    """
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger


@dataclass
class ConversionResult:
    """Represents the outcome of a single file conversion."""

    input_file: Path
    output_file: Optional[Path] = None
    rows: int = 0
    sheets: int = 0
    error: Optional[str] = None
    skipped: bool = False

    @classmethod
    def success(
        cls,
        input_file: Path,
        output_file: Path,
        rows: int,
        sheets: int = 1,
    ) -> "ConversionResult":
        return cls(
            input_file=input_file,
            output_file=output_file,
            rows=rows,
            sheets=sheets,
        )

    @classmethod
    def failure(cls, input_file: Path, error: str) -> "ConversionResult":
        return cls(input_file=input_file, error=error)

    @classmethod
    def skipped_file(cls, input_file: Path, reason: str) -> "ConversionResult":
        return cls(input_file=input_file, error=reason, skipped=True)

    @property
    def ok(self) -> bool:
        return not self.skipped and self.error is None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok

    def describe(self) -> str:
        if self.skipped:
            return f"{self.input_file.name}: skipped ({self.error})"
        if self.ok:
            row_text = f"{self.rows:,}行" if self.rows else "0行"
            output_name = self.output_file.name if self.output_file else "N/A"
            return f"{self.input_file.name} → {output_name} ({row_text})"
        return f"{self.input_file.name}: {self.error}"

    def log_to(self, logger: logging.Logger) -> None:
        if self.skipped:
            logger.debug("  - %s", self.describe())
        elif self.ok:
            logger.info("  - %s", self.describe())
        else:
            logger.error("  - %s", self.describe())


def summarize_results(results: Iterable[ConversionResult], logger: logging.Logger) -> None:
    """Emit a consolidated summary for a batch of conversion results."""
    results = list(results)
    success = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]

    total_rows = sum(r.rows for r in success)

    logger.info("=" * 60)
    logger.info("処理結果サマリー")
    logger.info("=" * 60)
    logger.info("成功: %dファイル", len(success))
    logger.info("失敗: %dファイル", len(failures))
    logger.info("スキップ: %dファイル", len(skipped))
    logger.info("総行数: %s行", f"{total_rows:,}")

    if success:
        logger.info("-" * 60)
        logger.info("✓ 成功したファイル:")
        for result in success:
            result.log_to(logger)

    if skipped:
        logger.info("-" * 60)
        logger.info("⏭ スキップされたファイル:")
        for result in skipped:
            result.log_to(logger)

    if failures:
        logger.info("-" * 60)
        logger.info("✗ 失敗したファイル:")
        for result in failures:
            result.log_to(logger)


__all__ = ["configure_logging", "ConversionResult", "summarize_results"]
