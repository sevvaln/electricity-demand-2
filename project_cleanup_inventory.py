from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
PROPOSED_ROOT = PROJECT_ROOT / "proje2_clean"
INVENTORY_PATH = PROJECT_ROOT / "project_file_inventory.csv"
MOVE_PLAN_PATH = PROJECT_ROOT / "cleanup_move_plan.csv"
SUMMARY_PATH = PROJECT_ROOT / "cleanup_dry_run_summary.txt"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


FINAL_EXACT_KEEP = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "population_weighted_hdd_cdd.py",
    "talep_tahmin_tek_dosya.py",
    "xgboost_population_weighted_pipeline.py",
    "xgboost_population_weighted_diagnostics.py",
    "xgboost_population_weighted_spec_consistency.py",
    "xgboost_table3_light_validation.py",
    "xgboost_nonrecursive_historical_design_check.py",
    "xgboost_nonrecursive_historical_overfitting_audit.py",
    "xgboost_nonrecursive_historical_model_c_final.py",
    "xgboost_nonrecursive_historical_shap_analysis.py",
    "table3_medium_horizon_model.py",
    "forecast_2026_generation.py",
    "outputs/reports/population_weighted_hourly_hdd_cdd.csv",
    "outputs/reports/population_weighted_methodology_note.txt",
    "outputs/reports/population_weighted_match_validation.txt",
    "outputs/reports/population_weighted_weight_validation.txt",
    "outputs/reports/population_weighted_year_coverage.csv",
    "outputs/reports/population_weighted_yearly_adjusted_weights.csv",
    "outputs/reports/population_weighted_matched_cities.csv",
    "outputs/reports/population_weighted_city_weights.csv",
    "outputs/xgboost/table3_medium_horizon_metrics.csv",
    "outputs/xgboost/table3_medium_horizon_monthly_validation.csv",
    "outputs/xgboost/table3_medium_horizon_leakage_audit.txt",
    "outputs/xgboost/table3_medium_horizon_feature_importance.csv",
    "outputs/xgboost/table3_medium_horizon_interpretation.txt",
    "outputs/figures/actual_vs_forecast_2025.png",
    "outputs/figures/monthly_rmse_plot.png",
    "outputs/figures/residual_plot.png",
    "outputs/figures/feature_importance_plot.png",
    "outputs/xgboost/table3_sarima_guided_population_weighted_metrics.json",
    "outputs/xgboost/table3_sarima_guided_population_weighted_predictions_2025.csv",
    "outputs/xgboost/table3_sarima_guided_population_weighted_train_metrics.json",
    "outputs/xgboost/table3_light_validation_population_weighted.txt",
    "outputs/xgboost/table3_light_validation_population_weighted_metrics.csv",
    "outputs/xgboost/table8_nonrecursive_historical_population_weighted_metrics.json",
    "outputs/xgboost/table8_nonrecursive_historical_population_weighted_predictions_2025.csv",
    "outputs/xgboost/table8_nonrecursive_historical_population_weighted_feature_importance.csv",
    "outputs/xgboost/table8_nonrecursive_historical_population_weighted_methodology_note.txt",
    "outputs/xgboost/table8_historical_profile_design_check_2025.csv",
    "outputs/xgboost/table8_historical_profile_design_check_2025.txt",
    "outputs/xgboost/nonrecursive_historical_overfitting_audit.csv",
    "outputs/xgboost/nonrecursive_historical_regularization_study.csv",
    "outputs/xgboost/nonrecursive_historical_monthly_stability.csv",
    "outputs/figures/shap_summary_final_model.png",
    "outputs/figures/shap_bar_final_model.png",
    "outputs/xgboost/shap_values_mean_abs_final_model.csv",
    "outputs/figures/shap_dependence_weighted_CDD.png",
    "outputs/figures/shap_dependence_weighted_HDD.png",
    "outputs/figures/shap_dependence_historical_profile.png",
    "outputs/figures/shap_dependence_calendar_effects.png",
    "outputs/xgboost/shap_final_model_interpretation.txt",
    "outputs/xgboost/table3_forecast_2026.csv",
    "outputs/xgboost/table3_forecast_2026_methodology_note.txt",
    "outputs/xgboost/table8_nonrecursive_historical_population_weighted_forecast_2026.csv",
    "outputs/xgboost/table8_forecast_2026_methodology_note.txt",
    "outputs/xgboost/table8_forecast_2026_feature_generation_audit.txt",
    "outputs/xgboost/forecast_2026_model_comparison.txt",
    "outputs/figures/forecast_2026_full_year_plot.png",
    "outputs/figures/forecast_2026_monthly_profile_plot.png",
    "outputs/figures/forecast_2026_seasonal_pattern_plot.png",
    "outputs/figures/forecast_2026_table3_vs_table8_comparison_plot.png",
    "outputs/xgboost/xgboost_proje_ozeti_tr.md",
}

ROLLING_EXPERIMENT_KEEP_ARCHIVE = {
    "table3_2025_rolling_backtest.py",
    "outputs/xgboost/table3_2025_rolling_backtest_metrics.csv",
    "outputs/xgboost/table3_2025_rolling_vs_recursive_comparison.csv",
    "outputs/xgboost/table3_2025_monthly_validation.csv",
    "outputs/xgboost/table3_2025_rolling_backtest_interpretation.txt",
    "outputs/figures/table3_2025_actual_vs_forecasts.png",
    "outputs/figures/table3_2025_monthly_rmse_comparison.png",
    "outputs/figures/table3_2025_monthly_error_profile.png",
    "outputs/figures/table3_2025_residual_comparison.png",
    "outputs/xgboost/table3_forecast_2026_full_year_recursive_previous.csv",
    "outputs/xgboost/table3_forecast_2026_rolling_vs_full_year_recursive.csv",
    "outputs/xgboost/table3_forecast_2026_rolling_vs_full_year_recursive.txt",
}

VALIDATION_SCRIPT_KEEP = {
    "table8_long_term_comparison.py",
    "xgboost_nonrecursive_historical_leakage_audit.py",
    "xgboost_nonrecursive_historical_ablation_study.py",
    "xgboost_population_weighted_nonrecursive_historical_experiment.py",
}

ARCHIVE_SCRIPT_PATTERNS = (
    "xgboost_notebook_2025.py",
    "xgboost_test_2025.py",
    "xgboost_recursive_test_2025.py",
    "xgboost_train.py",
)


def category_for_kept_exact(rel_path: str) -> tuple[str, str]:
    suffix = Path(rel_path).suffix.lower()
    if suffix == ".py":
        return "final_code", "explicit final script or reproducibility code"
    if rel_path.startswith("data/"):
        return "final_data", "explicit final input dataset"
    if rel_path.startswith("outputs/figures/"):
        return "final_figure", "explicit final figure requested for review"
    if rel_path.startswith("outputs/reports/"):
        if "validation" in rel_path or "coverage" in rel_path or "weights" in rel_path or "matched" in rel_path:
            return "validation_artifact", "population-weighted validation artifact kept for reproducibility"
        return "final_report", "final report-style artifact kept for review"
    if rel_path.startswith("outputs/xgboost/"):
        if suffix in {".txt", ".md"}:
            return "final_report", "final report / methodology note / project summary"
        if suffix in {".png"}:
            return "final_figure", "final figure kept for reviewer-facing package"
        if "validation" in rel_path or "design_check" in rel_path or "overfitting" in rel_path or "regularization" in rel_path or "stability" in rel_path or "light_validation" in rel_path or "shap_values" in rel_path:
            return "validation_artifact", "validation / robustness / interpretability artifact"
        return "final_output", "explicit final output kept for reviewer-facing package"
    return "final_output", "explicit final artifact kept"


def proposed_path(rel_path: str, category: str, action: str) -> str:
    path = Path(rel_path)
    name = path.name
    if action == "candidate_for_deletion":
        return ""
    if action == "manual_review":
        return f"proje2_clean/archive/manual_review/{name}"
    if action == "move_to_archive":
        if rel_path in ROLLING_EXPERIMENT_KEEP_ARCHIVE or "rolling" in rel_path or "recursive" in rel_path:
            return f"proje2_clean/archive/experiments/table3_rolling_month_ahead/{name}"
        if "non_weighted" in rel_path or "fully_blind_broad_future_forecast" in rel_path:
            return f"proje2_clean/archive/experiments/old_non_weighted_results/{name}"
        return f"proje2_clean/archive/experiments/discarded_trials/{name}"
    if category == "final_code":
        if "population_weighted" in name or name == "talep_tahmin_tek_dosya.py":
            return f"proje2_clean/src/01_population_weighted_weather/{name}"
        if "table3" in name:
            return f"proje2_clean/src/02_table3_medium_horizon/{name}"
        if "table8" in name or "nonrecursive" in name or "shap" in name:
            return f"proje2_clean/src/03_table8_nonrecursive_historical/{name}"
        if "forecast_2026" in name:
            return f"proje2_clean/src/04_forecast_2026/{name}"
        return f"proje2_clean/src/utils/{name}"
    if category == "final_data":
        if rel_path.startswith("data/raw/"):
            return f"proje2_clean/data/raw/{path.relative_to('data/raw').as_posix()}"
        if rel_path.startswith("data/processed/"):
            return f"proje2_clean/data/processed/{path.relative_to('data/processed').as_posix()}"
        return f"proje2_clean/data/external/{name}"
    if category == "final_figure":
        return f"proje2_clean/outputs/figures/{name}"
    if category == "final_report":
        return f"proje2_clean/outputs/reports/{name}"
    if category == "validation_artifact":
        return f"proje2_clean/outputs/validation/{name}"
    return f"proje2_clean/outputs/predictions/{name}"


def sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Record:
    file_path: str
    file_name: str
    extension: str
    size_mb: float
    modified_date: str
    category_guess: str
    keep_recommendation: str
    reason: str
    proposed_new_path: str
    action: str


def iter_project_files() -> Iterable[Path]:
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_path = rel(path)
        parts = rel_path.split("/")
        if parts[0] in {".git", "proje2_clean"}:
            continue
        if path.name in {INVENTORY_PATH.name, MOVE_PLAN_PATH.name, SUMMARY_PATH.name}:
            continue
        yield path


def build_duplicate_map(paths: list[Path]) -> dict[str, list[Path]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        by_size[path.stat().st_size].append(path)
    duplicates: dict[str, list[Path]] = {}
    for same_size_paths in by_size.values():
        if len(same_size_paths) < 2:
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in same_size_paths:
            hashes[sha1(path)].append(path)
        for digest, dup_paths in hashes.items():
            if len(dup_paths) > 1:
                duplicates[digest] = dup_paths
    return duplicates


def classify(path: Path, duplicate_map: dict[str, list[Path]]) -> Record:
    rel_path = rel(path)
    ext = path.suffix.lower()
    size_mb = round(path.stat().st_size / 1_000_000, 4)
    modified_date = path.stat().st_mtime
    modified_str = __import__("datetime").datetime.fromtimestamp(modified_date).isoformat(sep=" ", timespec="seconds")

    category = "unknown_review_manually"
    action = "manual_review"
    reason = "no safe automatic rule matched; review manually"

    if rel_path in FINAL_EXACT_KEEP:
        category, reason = category_for_kept_exact(rel_path)
        action = "keep_in_final"
    elif rel_path in ROLLING_EXPERIMENT_KEEP_ARCHIVE:
        category = "archived_experiment"
        action = "move_to_archive"
        reason = "important rolling / recursive experiment kept for methodology history"
    elif rel_path in VALIDATION_SCRIPT_KEEP:
        category = "validation_artifact"
        action = "keep_in_final"
        reason = "validation / audit script needed for reproducibility"
    elif any(part == "__pycache__" for part in Path(rel_path).parts) or ext == ".pyc":
        category = "temporary_or_cache"
        action = "candidate_for_deletion"
        reason = "Python cache artifact; safe deletion candidate"
    elif any(part == ".ipynb_checkpoints" for part in Path(rel_path).parts):
        category = "temporary_or_cache"
        action = "candidate_for_deletion"
        reason = "notebook checkpoint; safe deletion candidate"
    elif path.stat().st_size == 0:
        category = "temporary_or_cache"
        action = "candidate_for_deletion"
        reason = "empty file"
    elif rel_path.startswith("data/raw/"):
        category = "final_data"
        action = "keep_in_final"
        reason = "raw source data required for reproducibility"
    elif rel_path.startswith("data/processed/analiz_verisi_") or rel_path == "data/processed/xgboost_train_2022_2024.csv":
        category = "final_data"
        action = "keep_in_final"
        reason = "processed input used by final forecasting workflow"
    elif rel_path.startswith("data/processed/"):
        category = "archived_experiment"
        action = "move_to_archive"
        reason = "older processed experiment dataset; not part of final reviewer-facing package"
    elif rel_path.startswith("outputs/models/"):
        category = "archived_experiment"
        action = "move_to_archive"
        reason = "saved model artifact from prior experiments or benchmarks; archive for reproducibility"
    elif rel_path.startswith("outputs/figures/"):
        category = "archived_experiment"
        action = "move_to_archive"
        reason = "figure not in final reviewer-facing set; keep in archive"
    elif rel_path.startswith("outputs/reports/"):
        if rel_path == "outputs/reports/population_weighted_province_hourly_hdd_cdd.csv":
            category = "archived_experiment"
            action = "move_to_archive"
            reason = "large province-hour intermediate file; preserve in archive, not reviewer-facing final package"
        else:
            category = "validation_artifact"
            action = "keep_in_final"
            reason = "population-weighted report or validation artifact worth preserving"
    elif rel_path.startswith("outputs/xgboost/"):
        name = path.name.lower()
        if "intramonth" in name or "broadscale" in name or "enhanced" in name or "stabilized" in name:
            category = "duplicate_or_superseded"
            action = "move_to_archive"
            reason = "older model family superseded by final population-weighted designs"
        elif "feature_generation_timeline_report" in name or "leakage" in name:
            category = "validation_artifact"
            action = "keep_in_final"
            reason = "leakage / feature-generation audit worth preserving"
        elif "notebook" in name or "recursive_test_2025_summary" in name or "test_2025_summary" in name or "test_hazirlik" in name:
            category = "duplicate_or_superseded"
            action = "move_to_archive"
            reason = "older notebook or test-prep artifact superseded by final curated outputs"
        elif "train_summary" in name or "training_summary" in name or "train_metrics" in name or "error_comparisons" in name or "xgboost_feature_importance" in name:
            category = "archived_experiment"
            action = "move_to_archive"
            reason = "older aggregate summary or generic artifact; archive for traceability"
        elif "fully_blind_broad_future_forecast" in name or "xgboost_backtest" in name or "xgboost_train_predictions_2022_2024" in name:
            category = "duplicate_or_superseded"
            action = "move_to_archive"
            reason = "older intermediate result superseded by final curated outputs"
        elif "population_weighted" in name or "table3" in name or "table8" in name or "shap" in name or "nonrecursive" in name:
            category = "archived_experiment"
            action = "move_to_archive"
            reason = "experiment / diagnostic output not in explicit final keep set"
        else:
            category = "unknown_review_manually"
            action = "manual_review"
            reason = "xgboost artifact not confidently classifiable"
    elif ext == ".py":
        if path.name in ARCHIVE_SCRIPT_PATTERNS or "experiment" in path.name.lower() or "ablation" in path.name.lower():
            category = "archived_experiment"
            action = "move_to_archive"
            reason = "older experiment script or superseded analysis code"
        else:
            category = "final_code"
            action = "keep_in_final"
            reason = "code file not obviously obsolete; preserve in final source tree"
    elif ext in {".md", ".txt"}:
        category = "final_report"
        action = "keep_in_final"
        reason = "documentation or human-readable note"

    digest = None
    if action != "candidate_for_deletion" and path.stat().st_size > 0 and path.stat().st_size < 50_000_000:
        try:
            digest = sha1(path)
        except OSError:
            digest = None
    if digest and digest in duplicate_map and len(duplicate_map[digest]) > 1:
        dup_group = sorted(rel(p) for p in duplicate_map[digest])
        canonical = dup_group[0]
        if rel_path != canonical and rel_path not in FINAL_EXACT_KEEP:
            category = "duplicate_or_superseded"
            action = "candidate_for_deletion"
            reason = f"exact duplicate of {canonical}"

    proposed = proposed_path(rel_path, category, action)
    return Record(
        file_path=str(path),
        file_name=path.name,
        extension=ext,
        size_mb=size_mb,
        modified_date=modified_str,
        category_guess=category,
        keep_recommendation=action,
        reason=reason,
        proposed_new_path=proposed,
        action=action,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_proposed_structure() -> None:
    dirs = [
        PROPOSED_ROOT / "data" / "raw",
        PROPOSED_ROOT / "data" / "processed",
        PROPOSED_ROOT / "data" / "external",
        PROPOSED_ROOT / "src" / "01_population_weighted_weather",
        PROPOSED_ROOT / "src" / "02_table3_medium_horizon",
        PROPOSED_ROOT / "src" / "03_table8_nonrecursive_historical",
        PROPOSED_ROOT / "src" / "04_forecast_2026",
        PROPOSED_ROOT / "src" / "utils",
        PROPOSED_ROOT / "outputs" / "metrics",
        PROPOSED_ROOT / "outputs" / "predictions",
        PROPOSED_ROOT / "outputs" / "figures",
        PROPOSED_ROOT / "outputs" / "reports",
        PROPOSED_ROOT / "outputs" / "validation",
        PROPOSED_ROOT / "archive" / "experiments" / "table3_rolling_month_ahead",
        PROPOSED_ROOT / "archive" / "experiments" / "table3_full_year_recursive",
        PROPOSED_ROOT / "archive" / "experiments" / "discarded_trials",
        PROPOSED_ROOT / "archive" / "experiments" / "old_non_weighted_results",
        PROPOSED_ROOT / "archive" / "manual_review",
        PROPOSED_ROOT / "docs",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    readme = PROPOSED_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# proje2_clean README (Draft)",
                "",
                "This directory is a dry-run proposal only. No files have been moved or deleted yet.",
                "",
                "## Project Objective",
                "Forecast hourly electricity demand for Turkey, using validated population-weighted weather inputs and XGBoost-based forecasting pipelines.",
                "",
                "## Final Model Structure",
                "- Final Table 3: medium-horizon, non-recursive, historical-profile XGBoost using population-weighted HDD/CDD.",
                "- Final Table 8: non-recursive historical-analog XGBoost using previous-year analog and grouped profile features.",
                "",
                "## Data Flow",
                "1. Raw consumption, temperature, and economic inputs are stored under `data/raw/`.",
                "2. Population-weighted weather series and processed analysis frames are stored under `data/processed/` and `outputs/reports/`.",
                "3. Final model scripts live under `src/`.",
                "4. Final metrics, predictions, figures, reports, and validation artifacts live under `outputs/`.",
                "5. Non-selected experiments are retained under `archive/experiments/`.",
                "",
                "## How to Reproduce Final Results",
                "- Population-weighted weather construction: `src/01_population_weighted_weather/`",
                "- Final Table 3 medium-horizon workflow: `src/02_table3_medium_horizon/`",
                "- Final Table 8 non-recursive historical workflow: `src/03_table8_nonrecursive_historical/`",
                "- Final 2026 forecast generation: `src/04_forecast_2026/`",
                "",
                "## Important Reviewer-Facing Outputs",
                "- Final Table 3 medium-horizon metrics, leakage audit, monthly validation, and feature importance",
                "- Final Table 8 metrics, design checks, overfitting/regularization outputs, and SHAP artifacts",
                "- Final 2026 forecast CSVs, methodology notes, comparison notes, and figures",
                "",
                "## Archived Folders",
                "- `archive/experiments/table3_rolling_month_ahead/`: operational rolling Table 3 experiments retained for methodology history",
                "- `archive/experiments/discarded_trials/`: superseded or exploratory model outputs",
                "- `archive/experiments/old_non_weighted_results/`: pre-population-weighted or superseded variants",
                "",
                "## Safety Note",
                "This is a dry-run cleanup proposal. Review `project_file_inventory.csv` and `cleanup_move_plan.csv` before moving or deleting anything.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    files = sorted(iter_project_files())
    duplicate_map = build_duplicate_map(files)
    records = [classify(path, duplicate_map) for path in files]

    inventory_rows = [
        {
            "file_path": record.file_path,
            "file_name": record.file_name,
            "extension": record.extension,
            "size_mb": record.size_mb,
            "modified_date": record.modified_date,
            "category_guess": record.category_guess,
            "keep_recommendation": record.keep_recommendation,
            "reason": record.reason,
        }
        for record in records
    ]
    write_csv(
        INVENTORY_PATH,
        inventory_rows,
        ["file_path", "file_name", "extension", "size_mb", "modified_date", "category_guess", "keep_recommendation", "reason"],
    )

    move_rows = [
        {
            "current_path": record.file_path,
            "proposed_new_path": record.proposed_new_path,
            "action": record.action,
            "reason": record.reason,
        }
        for record in records
    ]
    write_csv(
        MOVE_PLAN_PATH,
        move_rows,
        ["current_path", "proposed_new_path", "action", "reason"],
    )

    ensure_proposed_structure()

    total_size_mb = sum(path.stat().st_size for path in files) / 1_000_000
    candidate_delete_size_mb = sum(
        Path(record.file_path).stat().st_size for record in records if record.action == "candidate_for_deletion"
    ) / 1_000_000
    archive_move_size_mb = sum(
        Path(record.file_path).stat().st_size for record in records if record.action == "move_to_archive"
    ) / 1_000_000
    keep_in_final_size_mb = sum(
        Path(record.file_path).stat().st_size for record in records if record.action == "keep_in_final"
    ) / 1_000_000
    manual_review_size_mb = sum(
        Path(record.file_path).stat().st_size for record in records if record.action == "manual_review"
    ) / 1_000_000
    estimated_clean_final_size_mb = total_size_mb - candidate_delete_size_mb

    largest_20 = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:20]
    largest_20_lines = [
        f"- {rel(path)} | {path.stat().st_size / 1_000_000:.2f} MB"
        for path in largest_20
    ]

    manual_review_files = [record for record in records if record.action == "manual_review"]
    manual_review_preview = [
        f"- {Path(record.file_path).relative_to(PROJECT_ROOT).as_posix()} | {record.size_mb:.2f} MB | {record.reason}"
        for record in manual_review_files[:40]
    ]

    summary_lines = [
        "Project Cleanup Dry-Run Summary",
        "",
        "Important note: `.git/` metadata and `proje2_clean/` draft structure are excluded from the inventory because they are not reviewer-facing project artifacts.",
        "",
        f"Current total project size (inventory scope): {total_size_mb:.2f} MB",
        f"Estimated clean final size (after only candidate deletions): {estimated_clean_final_size_mb:.2f} MB",
        f"Reviewer-facing keep_in_final size: {keep_in_final_size_mb:.2f} MB",
        f"Archive move total size: {archive_move_size_mb:.2f} MB",
        f"Candidate deletion total size: {candidate_delete_size_mb:.2f} MB",
        f"Manual review total size: {manual_review_size_mb:.2f} MB",
        "",
        "Largest 20 files:",
        *largest_20_lines,
        "",
        f"Files needing manual review: {len(manual_review_files)}",
        *manual_review_preview,
        "",
        f"Inventory written to: {INVENTORY_PATH}",
        f"Move plan written to: {MOVE_PLAN_PATH}",
        f"Proposed clean structure draft created under: {PROPOSED_ROOT}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
