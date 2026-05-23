from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
MOVE_PLAN_V2_PATH = PROJECT_ROOT / "cleanup_move_plan_v2.csv"
REVIEWER_MANIFEST_PATH = PROJECT_ROOT / "reviewer_package_manifest.csv"
ARCHIVE_MANIFEST_PATH = PROJECT_ROOT / "archive_manifest.csv"
SUMMARY_V2_PATH = PROJECT_ROOT / "cleanup_v2_summary.txt"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


REVIEWER_EXACT = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "population_weighted_hdd_cdd.py",
    "talep_tahmin_tek_dosya.py",
    "xgboost_population_weighted_pipeline.py",
    "xgboost_population_weighted_diagnostics.py",
    "xgboost_population_weighted_spec_consistency.py",
    "xgboost_nonrecursive_historical_design_check.py",
    "xgboost_nonrecursive_historical_model_c_final.py",
    "xgboost_nonrecursive_historical_overfitting_audit.py",
    "xgboost_nonrecursive_historical_shap_analysis.py",
    "xgboost_table3_light_validation.py",
    "table3_medium_horizon_model.py",
    "forecast_2026_generation.py",
    "outputs/reports/population_weighted_hourly_hdd_cdd.csv",
    "outputs/reports/population_weighted_methodology_note.txt",
    "outputs/reports/population_weighted_match_validation.txt",
    "outputs/reports/population_weighted_weight_validation.txt",
    "outputs/reports/population_weighted_year_coverage.csv",
    "outputs/reports/population_weighted_yearly_adjusted_weights.csv",
    "data/processed/xgboost_train_2022_2024.csv",
    "data/processed/analiz_verisi_2025.csv",
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

EXPLICIT_REVIEWER_EXCLUDES = {
    "outputs/reports/population_weighted_province_hourly_hdd_cdd.csv",
}

SAFE_DELETE_PATTERNS = {
    "__pycache__",
    ".ipynb_checkpoints",
}


@dataclass
class Row:
    current_path: str
    package: str
    proposed_new_path: str
    action: str
    category: str
    size_mb: float
    reason: str
    compression_recommendation: str
    safe_delete_after_backup: str


def iter_files() -> Iterable[Path]:
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = rel(path)
        if relative.split("/")[0] in {".git", "proje2_clean"}:
            continue
        if path.name in {
            MOVE_PLAN_V2_PATH.name,
            REVIEWER_MANIFEST_PATH.name,
            ARCHIVE_MANIFEST_PATH.name,
            SUMMARY_V2_PATH.name,
        }:
            continue
        yield path


def classify(path: Path) -> Row:
    relative = rel(path)
    size_mb = round(path.stat().st_size / 1_000_000, 4)
    ext = path.suffix.lower()
    name = path.name.lower()
    parts = set(Path(relative).parts)

    if any(part in SAFE_DELETE_PATTERNS for part in parts) or ext == ".pyc":
        return Row(
            current_path=str(path),
            package="excluded",
            proposed_new_path="",
            action="candidate_for_deletion",
            category="temporary_or_cache",
            size_mb=size_mb,
            reason="cache or temporary artifact",
            compression_recommendation="not_needed",
            safe_delete_after_backup="yes",
        )

    if relative in REVIEWER_EXACT:
        return Row(
            current_path=str(path),
            package="reviewer_package",
            proposed_new_path=proposed_reviewer_path(relative),
            action="keep_in_reviewer_package",
            category=reviewer_category(relative),
            size_mb=size_mb,
            reason="explicitly selected final asset for external review",
            compression_recommendation="not_needed",
            safe_delete_after_backup="no",
        )

    if relative in EXPLICIT_REVIEWER_EXCLUDES:
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/large_intermediates/{Path(relative).name}",
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="large intermediate file; not needed in reviewer package",
            compression_recommendation="compress_or_external_backup",
            safe_delete_after_backup="no",
        )

    if relative.startswith("data/raw/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/data/raw/{Path(relative).relative_to('data/raw').as_posix()}",
            action="move_to_archive_only",
            category="final_data",
            size_mb=size_mb,
            reason="raw source data preserved for full archive but omitted from reviewer package for size and clarity",
            compression_recommendation="consider_external_backup_if_large",
            safe_delete_after_backup="no",
        )

    if relative.startswith("data/processed/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/data/processed/{Path(relative).relative_to('data/processed').as_posix()}",
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="processed dataset not in minimal reviewer package",
            compression_recommendation="consider_compression_if_large",
            safe_delete_after_backup="no",
        )

    if relative.startswith("outputs/models/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/models/{Path(relative).name}",
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="saved model artifact retained for full archive only",
            compression_recommendation="consider_compression_if_large",
            safe_delete_after_backup="no",
        )

    if relative.startswith("outputs/figures/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=archive_figure_path(relative),
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="figure not in curated reviewer package",
            compression_recommendation="not_needed",
            safe_delete_after_backup="no",
        )

    if relative.startswith("outputs/reports/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/reports/{Path(relative).name}",
            action="move_to_archive_only",
            category="validation_artifact",
            size_mb=size_mb,
            reason="report or validation artifact kept in full archive only",
            compression_recommendation="consider_compression_if_large" if size_mb > 10 else "not_needed",
            safe_delete_after_backup="no",
        )

    if relative.startswith("outputs/xgboost/"):
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=archive_xgboost_path(relative),
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason=archive_reason(relative),
            compression_recommendation="consider_compression_if_large" if size_mb > 5 else "not_needed",
            safe_delete_after_backup="no",
        )

    if ext == ".py":
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=archive_script_path(relative),
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="script preserved in full archive but not needed in reviewer package",
            compression_recommendation="not_needed",
            safe_delete_after_backup="no",
        )

    if ext in {".md", ".txt"}:
        return Row(
            current_path=str(path),
            package="archive_full",
            proposed_new_path=f"archive_full/docs/{Path(relative).name}",
            action="move_to_archive_only",
            category="archived_experiment",
            size_mb=size_mb,
            reason="documentation not part of minimal reviewer package",
            compression_recommendation="not_needed",
            safe_delete_after_backup="no",
        )

    return Row(
        current_path=str(path),
        package="archive_full",
        proposed_new_path=f"archive_full/misc/{Path(relative).name}",
        action="move_to_archive_only",
        category="unknown_review_manually",
        size_mb=size_mb,
        reason="defaulted to archive_full to avoid accidental loss",
        compression_recommendation="not_needed",
        safe_delete_after_backup="no",
    )


def reviewer_category(relative: str) -> str:
    if relative.endswith(".py"):
        return "final_code"
    if relative.startswith("data/"):
        return "final_data"
    if relative.startswith("outputs/figures/"):
        return "final_figure"
    if relative.endswith(".md") or relative.endswith(".txt"):
        return "final_report"
    if "validation" in relative or "design_check" in relative or "overfitting" in relative or "regularization" in relative or "stability" in relative or "shap_values" in relative:
        return "validation_artifact"
    return "final_output"


def proposed_reviewer_path(relative: str) -> str:
    path = Path(relative)
    name = path.name
    if relative == ".gitignore":
        return "reviewer_package/.gitignore"
    if relative == "README.md":
        return "reviewer_package/README.md"
    if relative == "requirements.txt":
        return "reviewer_package/requirements.txt"
    if relative.startswith("data/processed/"):
        return f"reviewer_package/data/processed/{name}"
    if relative.startswith("outputs/reports/"):
        return f"reviewer_package/outputs/validation/{name}"
    if relative.startswith("outputs/xgboost/"):
        if name.endswith(".md") or name.endswith(".txt"):
            return f"reviewer_package/outputs/reports/{name}"
        if "validation" in name or "design_check" in name or "overfitting" in name or "regularization" in name or "stability" in name:
            return f"reviewer_package/outputs/validation/{name}"
        return f"reviewer_package/outputs/metrics/{name}"
    if relative.startswith("outputs/figures/"):
        return f"reviewer_package/outputs/figures/{name}"
    if relative.endswith(".py"):
        if "population_weighted" in name or name == "talep_tahmin_tek_dosya.py":
            return f"reviewer_package/src/01_population_weighted_weather/{name}"
        if "table3" in name:
            return f"reviewer_package/src/02_table3_medium_horizon/{name}"
        if "table8" in name or "nonrecursive" in name or "shap" in name:
            return f"reviewer_package/src/03_table8_nonrecursive_historical/{name}"
        if "forecast_2026" in name:
            return f"reviewer_package/src/04_forecast_2026/{name}"
        return f"reviewer_package/src/utils/{name}"
    return f"reviewer_package/docs/{name}"


def archive_script_path(relative: str) -> str:
    name = Path(relative).name
    lower = name.lower()
    if "table3_2025_rolling" in lower or "rolling" in lower:
        return f"archive_full/experiments/table3_rolling_month_ahead/{name}"
    if "recursive" in lower:
        return f"archive_full/experiments/table3_full_year_recursive/{name}"
    if "notebook" in lower or "test_2025" in lower or "train.py" in lower:
        return f"archive_full/experiments/discarded_trials/{name}"
    return f"archive_full/experiments/other_scripts/{name}"


def archive_figure_path(relative: str) -> str:
    name = Path(relative).name.lower()
    if "table3_2025" in name:
        return f"archive_full/experiments/table3_rolling_month_ahead/{Path(relative).name}"
    if "broadscale" in name or "enhanced" in name or "stabilized" in name or "january" in name:
        return f"archive_full/experiments/discarded_trials/{Path(relative).name}"
    return f"archive_full/figures/{Path(relative).name}"


def archive_xgboost_path(relative: str) -> str:
    name = Path(relative).name.lower()
    original_name = Path(relative).name
    if "rolling" in name or "recursive" in name:
        return f"archive_full/experiments/table3_rolling_month_ahead/{original_name}"
    if "broadscale" in name or "enhanced" in name or "stabilized" in name or "fully_blind_broad_future_forecast" in name:
        return f"archive_full/experiments/discarded_trials/{original_name}"
    if "train_predictions" in name:
        return f"archive_full/large_intermediates/train_predictions/{original_name}"
    if "notebook" in name:
        return f"archive_full/experiments/discarded_trials/{original_name}"
    return f"archive_full/xgboost_misc/{original_name}"


def archive_reason(relative: str) -> str:
    name = Path(relative).name.lower()
    if "train_predictions" in name:
        return "large train-prediction intermediate; omit from reviewer package"
    if "broadscale" in name or "enhanced" in name or "stabilized" in name:
        return "discarded or superseded experiment family"
    if "rolling" in name or "recursive" in name:
        return "important methodology-history experiment retained in archive"
    if "notebook" in name:
        return "older notebook-derived artifact superseded by final curated outputs"
    return "non-final xgboost artifact retained in full archive"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [classify(path) for path in sorted(iter_files())]

    move_rows = [
        {
            "current_path": row.current_path,
            "proposed_new_path": row.proposed_new_path,
            "action": row.action,
            "package": row.package,
            "category": row.category,
            "size_mb": row.size_mb,
            "reason": row.reason,
            "compression_recommendation": row.compression_recommendation,
            "safe_delete_after_backup": row.safe_delete_after_backup,
        }
        for row in rows
    ]
    write_csv(
        MOVE_PLAN_V2_PATH,
        [
            "current_path",
            "proposed_new_path",
            "action",
            "package",
            "category",
            "size_mb",
            "reason",
            "compression_recommendation",
            "safe_delete_after_backup",
        ],
        move_rows,
    )

    reviewer_rows = [
        {
            "current_path": row.current_path,
            "proposed_new_path": row.proposed_new_path,
            "category": row.category,
            "size_mb": row.size_mb,
            "reason": row.reason,
        }
        for row in rows
        if row.package == "reviewer_package"
    ]
    write_csv(
        REVIEWER_MANIFEST_PATH,
        ["current_path", "proposed_new_path", "category", "size_mb", "reason"],
        reviewer_rows,
    )

    archive_rows = [
        {
            "current_path": row.current_path,
            "proposed_new_path": row.proposed_new_path,
            "category": row.category,
            "size_mb": row.size_mb,
            "reason": row.reason,
            "compression_recommendation": row.compression_recommendation,
            "safe_delete_after_backup": row.safe_delete_after_backup,
        }
        for row in rows
        if row.package == "archive_full" or row.action == "candidate_for_deletion"
    ]
    write_csv(
        ARCHIVE_MANIFEST_PATH,
        [
            "current_path",
            "proposed_new_path",
            "category",
            "size_mb",
            "reason",
            "compression_recommendation",
            "safe_delete_after_backup",
        ],
        archive_rows,
    )

    reviewer_size = sum(row.size_mb for row in rows if row.package == "reviewer_package")
    archive_size = sum(row.size_mb for row in rows if row.package == "archive_full")
    excluded_rows = [row for row in rows if row.package != "reviewer_package"]
    compression_rows = [
        row for row in rows if row.compression_recommendation in {"compress_or_external_backup", "consider_external_backup_if_large", "consider_compression_if_large"}
    ]
    delete_after_backup_rows = [row for row in rows if row.safe_delete_after_backup == "yes"]

    largest_archive = sorted(
        [row for row in rows if row.package == "archive_full"],
        key=lambda r: r.size_mb,
        reverse=True,
    )[:20]

    summary_lines = [
        "Cleanup Plan V2 Summary",
        "",
        "No files were moved or deleted. This is a planning-only deliverable.",
        "",
        f"reviewer_package_size_mb: {reviewer_size:.2f}",
        f"archive_full_size_mb: {archive_size:.2f}",
        f"files_excluded_from_reviewer_package: {len(excluded_rows)}",
        f"files_recommended_for_compression: {len(compression_rows)}",
        f"files_safe_to_delete_after_backup: {len(delete_after_backup_rows)}",
        "",
        "Largest archive_full candidates:",
        *[
            f"- {Path(row.current_path).relative_to(PROJECT_ROOT).as_posix()} | {row.size_mb:.2f} MB | {row.compression_recommendation}"
            for row in largest_archive
        ],
        "",
        "Files recommended for compression or external backup:",
        *[
            f"- {Path(row.current_path).relative_to(PROJECT_ROOT).as_posix()} | {row.size_mb:.2f} MB | {row.reason}"
            for row in sorted(compression_rows, key=lambda r: r.size_mb, reverse=True)[:40]
        ],
        "",
        "Files safe to delete after backup:",
        *[
            f"- {Path(row.current_path).relative_to(PROJECT_ROOT).as_posix()} | {row.size_mb:.4f} MB | {row.reason}"
            for row in delete_after_backup_rows[:50]
        ],
        "",
        "Suspicious-path fix note:",
        "- `.gitignore` is now mapped to `reviewer_package/.gitignore`, not under outputs.",
        "",
        f"Generated at: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        f"Move plan: {MOVE_PLAN_V2_PATH}",
        f"Reviewer manifest: {REVIEWER_MANIFEST_PATH}",
        f"Archive manifest: {ARCHIVE_MANIFEST_PATH}",
    ]
    SUMMARY_V2_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
