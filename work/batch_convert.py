#!/usr/bin/env python3
"""
Excel to CSV batch conversion utility for the work pipeline.

This refactored version keeps the original behaviour while improving
structure, logging, and error handling. Key features:
  * Format detection with optional structure analysis in debug mode
  * Automatic wide → long conversion for POS workbooks
  * Graceful fallbacks to simple and multi-sheet conversions
  * Optional multiprocessing
  * Post-processing hooks reused by the debug pipeline
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from work_utils import ConversionResult, configure_logging, summarize_results


ID_KEYWORDS = ("店舗", "フェイス", "商品", "分類")
DATE_TOKEN_PATTERN = re.compile(r"(\d{4}/\d{1,2}/\d{1,2})")
DISPLAY_BREAK = "=" * 60


@dataclass(frozen=True)
class ConversionTask:
    """Serializable payload for processing a single Excel workbook."""

    input_file: Path
    output_dir: Path
    use_wide_to_long: bool
    analyze: bool
    debug: bool


class ExcelConverter:
    """Handle conversion of one Excel workbook into CSV."""

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        *,
        use_wide_to_long: bool,
        analyze: bool,
        debug: bool,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = output_dir / f"{input_file.stem}.csv"
        self.use_wide_to_long = use_wide_to_long
        self.analyze = analyze
        self.debug = debug
        self.logger = logger or configure_logging(
            debug, name=f"work.batch_convert.{input_file.stem}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def convert(self) -> ConversionResult:
        """Convert the workbook to CSV and return the outcome."""
        if self.input_file.name.startswith("~$"):
            return ConversionResult.skipped_file(
                self.input_file, "Excel一時ファイル（スキップ）"
            )

        file_format = self._detect_file_format()

        if self.use_wide_to_long and file_format == "pos":
            result = self._convert_with_wide_to_long()
            if result.ok:
                return result
            self.logger.info("シンプル変換にフォールバック: %s", self.input_file.name)
            return self._convert_simple_fallback()

        return self._convert_smart()

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #
    def _convert_smart(self) -> ConversionResult:
        """Attempt conversion with header inference and wide → long reshape."""
        try:
            self.logger.info("処理開始: %s", self.input_file.name)

            if self.analyze:
                self._analyze_structure()

            df = pd.read_excel(self.input_file, header=None)
            header_idx = self._find_header_row(df)

            if header_idx == -1:
                self.logger.warning(
                    "ヘッダーが検出できませんでした。フォールバックを実行します"
                )
                return self._convert_simple_fallback()

            date_idx = self._find_date_row(df, header_idx)
            header_row = df.iloc[header_idx]
            id_columns = self._infer_id_columns(header_row)

            if not id_columns:
                self.logger.warning(
                    "ID列が検出できませんでした。フォールバックを実行します"
                )
                return self._convert_simple_fallback()

            result_df = self._convert_wide_to_long(df, header_idx, date_idx, id_columns)

            if result_df.empty:
                self.logger.warning("ロング形式への変換結果が空でした。フォールバックします")
                return self._convert_simple()

            result_df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
            self.logger.info(
                "✓ 完了: %s (%s行)",
                self.output_file.name,
                f"{len(result_df):,}",
            )
            return ConversionResult.success(
                self.input_file, self.output_file, len(result_df)
            )

        except Exception as exc:
            self._log_exception(f"スマート変換エラー [{self.input_file.name}]", exc)

        return self._convert_simple_fallback()

    def _convert_simple(self) -> ConversionResult:
        """Fallback that preserves the first sheet with inferred headers."""
        try:
            df = pd.read_excel(self.input_file)
            df.to_csv(self.output_file, index=False, encoding="utf-8-sig")
            self.logger.info(
                "✓ 完了（シンプル）: %s (%s行)",
                self.output_file.name,
                f"{len(df):,}",
            )
            return ConversionResult.success(
                self.input_file, self.output_file, len(df)
            )
        except Exception as exc:
            self._log_exception(
                f"シンプル変換でエラーが発生しました [{self.input_file.name}]", exc
            )
            return self._convert_simple_fallback()

    def _convert_simple_fallback(self) -> ConversionResult:
        """
        Final fallback that concatenates every non-empty sheet with the sheet name.
        """
        try:
            xlsx = pd.ExcelFile(self.input_file)
            frames: List[pd.DataFrame] = []

            for sheet_name in xlsx.sheet_names:
                frame = xlsx.parse(sheet_name=sheet_name)
                if frame.empty:
                    continue
                frame.insert(0, "シート名", sheet_name)
                frames.append(frame)

            if not frames:
                return ConversionResult.failure(
                    self.input_file, "有効なデータが見つかりません"
                )

            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(self.output_file, index=False, encoding="utf-8-sig")
            self.logger.info(
                "✓ 完了（フォールバック）: %s (%s行)",
                self.output_file.name,
                f"{len(combined):,}",
            )
            return ConversionResult.success(
                self.input_file, self.output_file, len(combined), sheets=len(frames)
            )

        except Exception as exc:
            self._log_exception(
                f"フォールバック変換でエラーが発生しました [{self.input_file.name}]",
                exc,
            )
            return ConversionResult.failure(
                self.input_file, f"フォールバック変換エラー: {exc}"
            )

    def _convert_with_wide_to_long(self) -> ConversionResult:
        """Invoke wide_to_long.py for POS data."""
        script = Path(__file__).resolve().parent / "wide_to_long.py"
        if not script.exists():
            self.logger.error("wide_to_long.pyが見つかりません")
            return ConversionResult.failure(
                self.input_file, "wide_to_long.pyが見つかりません"
            )

        command = [
            sys.executable,
            str(script),
            str(self.input_file),
            str(self.output_file),
            "--skip-weather",
            "--no-gpu",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=script.parent,
                check=False,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip() or "unknown error"
                self.logger.error("wide_to_long.py実行エラー: %s", stderr)
                return ConversionResult.failure(
                    self.input_file,
                    f"wide_to_long.py実行エラー (exit {result.returncode})",
                )

            df = pd.read_csv(self.output_file)
            self.logger.info(
                "✓ 完了（wide_to_long）: %s (%s行)",
                self.output_file.name,
                f"{len(df):,}",
            )
            return ConversionResult.success(
                self.input_file, self.output_file, len(df)
            )

        except Exception as exc:
            self._log_exception("wide_to_long.pyの実行中にエラー", exc)
            return ConversionResult.failure(self.input_file, str(exc))

    # ------------------------------------------------------------------ #
    # Detection utilities
    # ------------------------------------------------------------------ #
    def _detect_file_format(self) -> str:
        """Attempt to classify the workbook format."""
        try:
            filename = self.input_file.name
            if "売上情報" in filename:
                return "sales"
            if "POS情報" in filename:
                return "pos"

            df = pd.read_excel(self.input_file, header=None, nrows=30)
            for idx in range(min(30, len(df))):
                row = df.iloc[idx]
                row_preview = " ".join(str(cell) for cell in row.tolist()[:10])
                if "店舗" in row_preview and "フェイス" in row_preview:
                    return "pos"
                if "店舗" in row_preview and "売上" in row_preview:
                    return "sales"
        except Exception as exc:
            self._log_exception("フォーマット検出に失敗しました", exc)
        return "unknown"

    def _analyze_structure(self) -> None:
        """Emit detailed insights about the workbook layout (debug helper)."""
        try:
            df = pd.read_excel(self.input_file, header=None, nrows=50)

            self.logger.info("📊 ファイル構造分析: %s", self.input_file.name)
            self.logger.info("  総行数: %d行（先頭50行のみ表示）", len(df))
            self.logger.info("  総列数: %d列", len(df.columns))
            self.logger.info("")

            self.logger.info("【先頭20行のプレビュー（左15列）】")
            for idx in range(min(20, len(df))):
                row_preview = df.iloc[idx, :15].tolist()
                truncated = [
                    str(value)[:20] if pd.notna(value) else "NaN" for value in row_preview
                ]
                self.logger.info("  行%2d: %s", idx, truncated)
            self.logger.info("")

            self.logger.info("【表頭（日付）の検出】")
            for idx in range(min(30, len(df))):
                row = df.iloc[idx]
                matches: List[Tuple[int, str]] = []
                for col_idx, value in enumerate(row):
                    if pd.notna(value):
                        match = DATE_TOKEN_PATTERN.search(str(value))
                        if match:
                            matches.append((col_idx, match.group(1)))
                if matches:
                    self.logger.info("  行%d: %d個の日付を検出", idx, len(matches))
                    for col_idx, token in matches[:5]:
                        self.logger.info("    列%d: %s", col_idx, token)
                    if len(matches) > 5:
                        self.logger.info("    ... 他%d個", len(matches) - 5)
                    break
            else:
                self.logger.warning("  日付が見つかりませんでした")
            self.logger.info("")

            self.logger.info("【表側（ヘッダー行）の検出】")
            for idx in range(min(30, len(df))):
                row = df.iloc[idx, :10].tolist()
                row_str = " ".join([str(value) for value in row if pd.notna(value)])
                if any(keyword in row_str for keyword in ("店舗", "商品", "フェイス", "分類")):
                    display = [
                        str(value)[:15] if pd.notna(value) else "NaN" for value in row[:10]
                    ]
                    self.logger.info("  行%d: ヘッダー候補を検出 %s", idx, display)
            self.logger.info("")

            self.logger.info("【表側データのサンプル（行8-18の左5列）】")
            for idx in range(8, min(18, len(df))):
                sample = df.iloc[idx, :5].tolist()
                truncated = [
                    str(value)[:20] if pd.notna(value) else "NaN" for value in sample
                ]
                self.logger.info("  行%d: %s", idx, truncated)
            self.logger.info("")

            self.logger.info("【左端列（第0列）のサンプル（行10-20）】")
            for idx in range(10, min(20, len(df))):
                value = df.iloc[idx, 0]
                if pd.notna(value):
                    self.logger.info("  行%d: %s", idx, str(value)[:50])
            self.logger.info("")

        except Exception as exc:
            self._log_exception("構造分析でエラーが発生しました", exc)

    @staticmethod
    def _find_header_row(df: pd.DataFrame) -> int:
        """Locate the row that contains the header columns."""
        for idx in range(min(30, len(df))):
            row = df.iloc[idx, :10].tolist()
            row_str = " ".join([str(value) for value in row if pd.notna(value)])
            if "店舗" in row_str and any(
                marker in row_str for marker in ("商品", "フェイス", "売上")
            ):
                return idx
        return -1

    @staticmethod
    def _find_date_row(df: pd.DataFrame, header_idx: int) -> int:
        """Identify the row that contains the date headers above the metrics."""
        for offset in range(1, 4):
            candidate = header_idx - offset
            if candidate < 0:
                continue
            row = df.iloc[candidate]
            hits = 0
            for value in row:
                if pd.notna(value) and DATE_TOKEN_PATTERN.search(str(value)):
                    hits += 1
            if hits:
                return candidate
        return -1

    @staticmethod
    def _infer_id_columns(header_row: pd.Series) -> List[str]:
        """Collect ID column names before the first metric column."""
        id_columns: List[str] = []
        for value in header_row:
            if pd.isna(value):
                continue
            header_str = str(value)
            if any(keyword in header_str for keyword in ID_KEYWORDS):
                id_columns.append(header_str)
                continue
            if id_columns:
                break
        return id_columns

    def _convert_wide_to_long(
        self,
        df: pd.DataFrame,
        header_idx: int,
        date_idx: int,
        id_columns: List[str],
    ) -> pd.DataFrame:
        """Transform the wide table into a long table with metrics per date."""
        header_row = df.iloc[header_idx]
        date_row = df.iloc[date_idx] if date_idx >= 0 else None
        data = df.iloc[header_idx + 1 :].copy()

        num_id_cols = len(id_columns)

        column_map: dict[int, Tuple[str, str]] = {}
        current_date: Optional[str] = None

        for col_idx in range(num_id_cols, len(header_row)):
            if date_row is not None and col_idx < len(date_row):
                cell_value = date_row.iloc[col_idx]
                if pd.notna(cell_value):
                    match = DATE_TOKEN_PATTERN.search(str(cell_value))
                    if match:
                        current_date = match.group(1)
                    else:
                        try:
                            current_date = pd.to_datetime(cell_value).strftime(
                                "%Y/%m/%d"
                            )
                        except Exception:
                            pass

            metric = header_row.iloc[col_idx]
            if pd.notna(metric) and current_date:
                column_map[col_idx] = (current_date, str(metric))

        for col_idx in range(num_id_cols, len(data.columns)):
            if col_idx in column_map:
                date_token, metric_name = column_map[col_idx]
                data.rename(columns={col_idx: (date_token, metric_name)}, inplace=True)
            else:
                data.rename(columns={col_idx: f"_drop_{col_idx}"}, inplace=True)

        keep_columns = [col for col in data.columns if not str(col).startswith("_drop_")]
        data = data[keep_columns]

        for col in id_columns:
            if col in data.columns:
                data[col] = data[col].replace("", pd.NA).ffill()

        if id_columns and id_columns[-1] in data.columns:
            data = data[data[id_columns[-1]].notna()]

        if id_columns:
            first_col = id_columns[0]
            if first_col in data.columns:
                data = data[
                    ~data[first_col].astype(str).str.contains(
                        "総合計|^合計$", na=False, regex=True
                    )
                ]

        for col in id_columns[1:-1]:
            if col in data.columns:
                data = data[~data[col].astype(str).str.fullmatch("合計", na=False)]

        id_vars = [col for col in id_columns if col in data.columns]
        value_columns = [col for col in data.columns if isinstance(col, tuple)]

        if not value_columns:
            return data

        melted = data.melt(
            id_vars=id_vars,
            value_vars=value_columns,
            var_name="date_metric",
            value_name="値",
        )

        melted[["日付", "指標"]] = pd.DataFrame(
            melted["date_metric"].tolist(), index=melted.index
        )
        melted = melted.drop(columns=["date_metric"])
        melted = melted.dropna(subset=["値"])

        result = (
            melted.pivot_table(
                index=id_vars + ["日付"],
                columns="指標",
                values="値",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        return result

    # ------------------------------------------------------------------ #
    # Logging helper
    # ------------------------------------------------------------------ #
    def _log_exception(self, message: str, exc: Exception) -> None:
        if self.debug:
            self.logger.exception(message)
        else:
            self.logger.error("%s: %s", message, exc)


# ---------------------------------------------------------------------- #
# Batch execution helpers
# ---------------------------------------------------------------------- #


def gather_excel_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.xlsx", "*.xls"):
        files.extend(sorted(input_dir.glob(pattern)))
    return sorted(files)


def _process_task(task: ConversionTask) -> ConversionResult:
    logger = configure_logging(
        task.debug, name=f"work.batch_convert.{Path(task.input_file).stem}"
    )
    converter = ExcelConverter(
        task.input_file,
        task.output_dir,
        use_wide_to_long=task.use_wide_to_long,
        analyze=task.analyze,
        debug=task.debug,
        logger=logger,
    )
    return converter.convert()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="work/input内の全Excelファイルを一括変換し、CSVとしてwork/outputに保存します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用例:
  python batch_convert.py
  python batch_convert.py --debug
  python batch_convert.py --use-wide-to-long
  python batch_convert.py --workers 4
""",
    )

    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード（詳細ログ）"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="特徴量付与をスキップ（CSV変換のみ実行）",
    )
    parser.add_argument(
        "--use-wide-to-long",
        action="store_true",
        help="POS情報ファイルにwide_to_long.pyを適用（実験的）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列処理ワーカー数（デフォルト: 1）",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="入力ディレクトリ（デフォルト: input）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="出力ディレクトリ（デフォルト: output）",
    )
    parser.add_argument(
        "--single-file",
        type=str,
        help="単一ファイルのみを処理（デバッグ用）",
    )

    return parser.parse_args(args=list(argv) if argv is not None else None)


def run_debug_steps(output_dir: Path, logger: logging.Logger) -> None:
    """Execute the auxiliary scripts that clean and enrich the 06-series files."""
    script_dir = Path(__file__).resolve().parent

    steps = [
        ("06_ファイルの統合", script_dir / "merge_converted_06.py"),
        ("06_ファイルのクリーンアップ", script_dir / "clean_06_data.py"),
    ]

    for description, script in steps:
        if not script.exists():
            logger.warning("%s: スクリプトが見つかりません (%s)", description, script)
            continue
        logger.info("%sを実行中...", description)
        try:
            subprocess.run(
                [sys.executable, str(script)],
                check=True,
                cwd=script.parent,
            )
            logger.info("✓ %sが完了しました", description)
        except subprocess.CalledProcessError as exc:
            logger.error("%sに失敗しました: %s", description, exc)

    enrich_script = script_dir / "enrich_features_v2.py"
    cleaned_file = output_dir / "06_cleaned_20250701_20250930.csv"
    enriched_file = output_dir / "06_final_enriched_20250701_20250930.csv"
    stores_file = script_dir / "stores.csv"
    past_year_file = (
        output_dir / "01_【売上情報】店別実績_20250903143116（20240901-20250831）.csv"
    )

    if enrich_script.exists() and cleaned_file.exists():
        logger.info("特徴量付与を実行中...")
        command = [
            sys.executable,
            str(enrich_script),
            str(cleaned_file),
            str(enriched_file),
            "--store-locations",
            str(stores_file),
            "--past-year-data",
            str(past_year_file),
        ]
        try:
            subprocess.run(command, check=True, cwd=enrich_script.parent)
            logger.info("✓ 特徴量付与が完了しました")
            logger.info("✅ 最終出力: %s", enriched_file)
        except subprocess.CalledProcessError as exc:
            logger.error("特徴量付与に失敗しました: %s", exc)
    else:
        logger.warning("特徴量付与スクリプトまたはクリーン済みファイルが見つかりません")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.debug, name="work.batch_convert")

    try:
        input_dir = Path(args.input_dir).resolve()
        output_dir = Path(args.output_dir).resolve()

        logger.info(DISPLAY_BREAK)
        logger.info("Excel → CSV バッチ変換")
        logger.info(DISPLAY_BREAK)
        logger.info("入力ディレクトリ: %s", input_dir)
        logger.info("出力ディレクトリ: %s", output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        analyze = args.debug
        tasks: List[ConversionTask] = []

        if args.single_file:
            target = input_dir / args.single_file
            if not target.exists():
                logger.error("ファイルが見つかりません: %s", target)
                sys.exit(1)
            tasks.append(
                ConversionTask(
                    target,
                    output_dir,
                    args.use_wide_to_long,
                    analyze,
                    args.debug,
                )
            )
        else:
            excel_files = gather_excel_files(input_dir)
            if not excel_files:
                logger.warning("Excelファイルが見つかりません: %s", input_dir)
                return
            logger.info("処理対象: %dファイル", len(excel_files))
            logger.info("-" * 60)
            for file_path in excel_files:
                tasks.append(
                    ConversionTask(
                        file_path,
                        output_dir,
                        args.use_wide_to_long,
                        analyze,
                        args.debug,
                    )
                )

        results: List[ConversionResult] = []

        if args.workers > 1 and len(tasks) > 1:
            logger.info("並列処理モード: %dワーカー", args.workers)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(_process_task, task): task for task in tasks
                }
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
        else:
            logger.info("シングルプロセスモード")
            for task in tasks:
                result = _process_task(task)
                results.append(result)

        summarize_results(results, logger)

        failures = [r for r in results if not r.ok and not r.skipped]
        if failures:
            logger.warning("%dファイルの処理に失敗しました", len(failures))
            sys.exit(1)

        logger.info("✅ 全ファイルの変換が完了しました")

        # 特徴量付与を実行（--skip-featuresが指定されていない場合）
        if not args.skip_features:
            logger.info("")
            logger.info(DISPLAY_BREAK)
            logger.info("ステップ2: 特徴量付与を開始")
            logger.info(DISPLAY_BREAK)
            run_debug_steps(output_dir, logger)
        else:
            logger.info("特徴量付与をスキップしました（--skip-features指定）")

    except Exception as exc:
        if args.debug:
            logger.exception("❌ エラー: %s", exc)
        else:
            logger.error("❌ エラー: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
