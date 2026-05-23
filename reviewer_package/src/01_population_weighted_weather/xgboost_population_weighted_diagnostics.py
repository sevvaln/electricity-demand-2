from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xgboost_population_weighted_pipeline import (
    ALL_YEARS,
    MODEL_CONFIGS,
    TEST_YEAR,
    TRAIN_YEARS,
    build_model_dataset,
    build_population_weighted_base_dataset,
    build_xgb_model,
    evaluate_predictions,
    invert_predictions,
    split_train_test,
    validate_dependencies_and_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent
XGBOOST_DIR = PROJECT_ROOT / "outputs" / "xgboost"

VALIDATION_REPORT_PATH = XGBOOST_DIR / "xgboost_population_weighted_model_validation.txt"
METRICS_CSV_PATH = XGBOOST_DIR / "xgboost_population_weighted_table3_table8_metrics.csv"
OVERFITTING_CSV_PATH = XGBOOST_DIR / "xgboost_population_weighted_overfitting_summary.csv"

TARGET_MODEL_KEYS = [
    "table3_sarima_guided_population_weighted",
    "table8_sarima_guided_population_weighted",
    "table8_fully_blind_population_weighted",
]

EXPECTED_FEATURES = {
    "table3_sarima_guided_population_weighted": [
        "weighted_HDD",
        "weighted_CDD",
        "night",
        "weekend",
        "log_consumption_lag_1h",
        "log_consumption_seasonal_diff_24h",
    ],
    "table8_sarima_guided_population_weighted": [
        "weighted_HDD",
        "weighted_CDD",
        "night",
        "weekend",
        "PMI_prev_month",
        "IR_prev_month",
        "log_consumption_lag_1h",
        "log_consumption_seasonal_diff_24h",
    ],
    "table8_fully_blind_population_weighted": [
        "weighted_HDD",
        "weighted_CDD",
        "night",
        "weekend",
        "PMI_prev_month",
        "IR_prev_month",
        "hour",
        "day_of_week",
        "month",
        "is_public_holiday",
        "is_holiday_window",
        "is_bridge_day",
    ],
}

FORBIDDEN_FULLY_BLIND_PATTERNS = [
    "consumption_lag_",
    "log_consumption_lag_1h",
    "log_consumption_seasonal_diff_24h",
    "consumption_roll_",
]


def get_target_configs() -> list[dict[str, object]]:
    config_map = {str(config["model_key"]): config for config in MODEL_CONFIGS}
    return [config_map[model_key] for model_key in TARGET_MODEL_KEYS]


def validate_merge(base_df: pd.DataFrame) -> dict[str, object]:
    expected_index = pd.date_range(
        start="2022-01-01 00:00:00",
        end="2025-12-31 23:00:00",
        freq="h",
    )
    actual_index = pd.DatetimeIndex(base_df["datetime"]).sort_values()

    missing_weighted_hdd = int(base_df["weighted_HDD"].isna().sum())
    missing_weighted_cdd = int(base_df["weighted_CDD"].isna().sum())
    complete_coverage = len(actual_index) == len(expected_index) and actual_index.equals(expected_index)

    merge_summary = {
        "merge_rows": len(base_df),
        "expected_rows": len(expected_index),
        "datetime_min": str(actual_index.min()),
        "datetime_max": str(actual_index.max()),
        "missing_weighted_HDD": missing_weighted_hdd,
        "missing_weighted_CDD": missing_weighted_cdd,
        "complete_datetime_coverage": bool(complete_coverage),
        "weighted_HDD_mean": float(base_df["weighted_HDD"].mean()),
        "weighted_CDD_mean": float(base_df["weighted_CDD"].mean()),
        "weighted_HDD_std": float(base_df["weighted_HDD"].std()),
        "weighted_CDD_std": float(base_df["weighted_CDD"].std()),
    }

    if {"HDD", "CDD", "weighted_HDD", "weighted_CDD"}.issubset(base_df.columns):
        merge_summary["hdd_weighted_hdd_corr"] = float(base_df["HDD"].corr(base_df["weighted_HDD"]))
        merge_summary["cdd_weighted_cdd_corr"] = float(base_df["CDD"].corr(base_df["weighted_CDD"]))
        merge_summary["hdd_mean_abs_diff"] = float(np.abs(base_df["HDD"] - base_df["weighted_HDD"]).mean())
        merge_summary["cdd_mean_abs_diff"] = float(np.abs(base_df["CDD"] - base_df["weighted_CDD"]).mean())

    return merge_summary


def validate_split(base_df: pd.DataFrame) -> dict[str, object]:
    train_df = base_df.loc[base_df["year"].isin(TRAIN_YEARS)].copy()
    test_df = base_df.loc[base_df["year"] == TEST_YEAR].copy()

    train_datetimes = set(pd.to_datetime(train_df["datetime"]))
    test_datetimes = set(pd.to_datetime(test_df["datetime"]))
    overlap = train_datetimes & test_datetimes

    return {
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_datetime_min": str(train_df["datetime"].min()),
        "train_datetime_max": str(train_df["datetime"].max()),
        "test_datetime_min": str(test_df["datetime"].min()),
        "test_datetime_max": str(test_df["datetime"].max()),
        "train_test_overlap_count": len(overlap),
        "no_train_test_overlap": len(overlap) == 0,
    }


def validate_feature_set(config: dict[str, object], available_columns: set[str]) -> dict[str, object]:
    model_key = str(config["model_key"])
    actual_features = list(config["feature_columns"])
    expected_features = [
        feature
        for feature in EXPECTED_FEATURES[model_key]
        if feature in available_columns or feature != "CUR_prev_month"
    ]
    missing_expected = [feature for feature in expected_features if feature not in actual_features]
    unexpected_features = [feature for feature in actual_features if feature not in expected_features]

    fully_blind_forbidden_found: list[str] = []
    if model_key == "table8_fully_blind_population_weighted":
        for feature in actual_features:
            if any(pattern in feature for pattern in FORBIDDEN_FULLY_BLIND_PATTERNS):
                fully_blind_forbidden_found.append(feature)

    status = "ok"
    if missing_expected or fully_blind_forbidden_found:
        status = "warning"

    return {
        "model_key": model_key,
        "label": config["label"],
        "status": status,
        "expected_features": expected_features,
        "actual_features": actual_features,
        "missing_expected": missing_expected,
        "unexpected_features": unexpected_features,
        "fully_blind_forbidden_found": fully_blind_forbidden_found,
    }


def fit_and_evaluate_models(base_df: pd.DataFrame, configs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    overfitting_rows: list[dict[str, object]] = []

    for config in configs:
        model_key = str(config["model_key"])
        target_column = str(config["target_column"])
        prediction_transform = str(config["prediction_transform"])
        dataset, feature_columns = build_model_dataset(base_df, config)
        train_df, test_df = split_train_test(dataset)

        train_overlap = set(pd.to_datetime(train_df["datetime"])) & set(pd.to_datetime(test_df["datetime"]))
        if train_overlap:
            raise ValueError(f"{model_key} icin train-test overlap bulundu.")

        model = build_xgb_model(model_key)
        model.fit(train_df[feature_columns], train_df[target_column])

        train_raw_predictions = model.predict(train_df[feature_columns])
        test_raw_predictions = model.predict(test_df[feature_columns])
        train_predictions = invert_predictions(prediction_transform, train_raw_predictions)
        test_predictions = invert_predictions(prediction_transform, test_raw_predictions)

        y_train = (
            np.square(train_df[target_column].to_numpy())
            if prediction_transform == "square_to_consumption"
            else train_df[target_column].to_numpy()
        )
        y_test = (
            np.square(test_df[target_column].to_numpy())
            if prediction_transform == "square_to_consumption"
            else test_df[target_column].to_numpy()
        )

        train_metrics = evaluate_predictions(y_train, train_predictions)
        test_metrics = evaluate_predictions(y_test, test_predictions)

        metrics_rows.append(
            {
                "model_key": model_key,
                "label": config["label"],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_R2": round(float(test_metrics["R2"]), 6),
                "test_MAE": round(float(test_metrics["MAE"]), 6),
                "test_RMSE": round(float(test_metrics["RMSE"]), 6),
                "test_MAPE": round(float(test_metrics["MAPE"]), 6),
            }
        )

        overfitting_rows.append(
            {
                "model_key": model_key,
                "label": config["label"],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_R2": round(float(train_metrics["R2"]), 6),
                "train_MAE": round(float(train_metrics["MAE"]), 6),
                "train_RMSE": round(float(train_metrics["RMSE"]), 6),
                "train_MAPE": round(float(train_metrics["MAPE"]), 6),
                "test_R2": round(float(test_metrics["R2"]), 6),
                "test_MAE": round(float(test_metrics["MAE"]), 6),
                "test_RMSE": round(float(test_metrics["RMSE"]), 6),
                "test_MAPE": round(float(test_metrics["MAPE"]), 6),
                "gap_R2_train_minus_test": round(float(train_metrics["R2"] - test_metrics["R2"]), 6),
                "gap_MAE_test_minus_train": round(float(test_metrics["MAE"] - train_metrics["MAE"]), 6),
                "gap_RMSE_test_minus_train": round(float(test_metrics["RMSE"] - train_metrics["RMSE"]), 6),
                "gap_MAPE_test_minus_train": round(float(test_metrics["MAPE"] - train_metrics["MAPE"]), 6),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    overfitting_df = pd.DataFrame(overfitting_rows)
    return metrics_df, overfitting_df


def write_validation_report(
    merge_summary: dict[str, object],
    split_summary: dict[str, object],
    feature_validations: list[dict[str, object]],
    metrics_df: pd.DataFrame,
    overfitting_df: pd.DataFrame,
) -> None:
    lines = [
        "XGBoost Population-Weighted Model Validation",
        "",
        "1. Data Merge Validation",
        f"- Rows after merge: {merge_summary['merge_rows']}",
        f"- Expected rows: {merge_summary['expected_rows']}",
        f"- Datetime min: {merge_summary['datetime_min']}",
        f"- Datetime max: {merge_summary['datetime_max']}",
        f"- Missing weighted_HDD: {merge_summary['missing_weighted_HDD']}",
        f"- Missing weighted_CDD: {merge_summary['missing_weighted_CDD']}",
        f"- Complete datetime coverage: {merge_summary['complete_datetime_coverage']}",
        f"- weighted_HDD mean/std: {merge_summary['weighted_HDD_mean']:.6f} / {merge_summary['weighted_HDD_std']:.6f}",
        f"- weighted_CDD mean/std: {merge_summary['weighted_CDD_mean']:.6f} / {merge_summary['weighted_CDD_std']:.6f}",
    ]
    if "hdd_weighted_hdd_corr" in merge_summary:
        lines.extend(
            [
                f"- HDD vs weighted_HDD correlation: {merge_summary['hdd_weighted_hdd_corr']:.6f}",
                f"- CDD vs weighted_CDD correlation: {merge_summary['cdd_weighted_cdd_corr']:.6f}",
                f"- HDD mean absolute difference: {merge_summary['hdd_mean_abs_diff']:.6f}",
                f"- CDD mean absolute difference: {merge_summary['cdd_mean_abs_diff']:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "2. Train/Test Split Validation",
            f"- Train years: {split_summary['train_years']}",
            f"- Test year: {split_summary['test_year']}",
            f"- Train rows: {split_summary['train_rows']}",
            f"- Test rows: {split_summary['test_rows']}",
            f"- Train datetime range: {split_summary['train_datetime_min']} to {split_summary['train_datetime_max']}",
            f"- Test datetime range: {split_summary['test_datetime_min']} to {split_summary['test_datetime_max']}",
            f"- Train/Test overlap count: {split_summary['train_test_overlap_count']}",
            f"- No train/test overlap: {split_summary['no_train_test_overlap']}",
            "",
            "3. Feature-Set Validation",
        ]
    )

    for result in feature_validations:
        lines.extend(
            [
                f"- {result['label']} [{result['status']}]",
                f"  expected: {result['expected_features']}",
                f"  actual: {result['actual_features']}",
                f"  missing_expected: {result['missing_expected']}",
                f"  unexpected_features: {result['unexpected_features']}",
                f"  fully_blind_forbidden_found: {result['fully_blind_forbidden_found']}",
            ]
        )

    lines.extend(
        [
            "",
            "4. Baseline Test Metrics",
            metrics_df.to_string(index=False),
            "",
            "5. Overfitting Summary",
            overfitting_df.to_string(index=False),
            "",
            "6. Notes",
            "- Hyperparameter tuning was not run.",
            "- This script checks correctness and leakage-related feature logic, not performance optimization.",
        ]
    )

    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    validate_dependencies_and_files()

    base_df, _ = build_population_weighted_base_dataset(ALL_YEARS)
    merge_summary = validate_merge(base_df)
    split_summary = validate_split(base_df)
    configs = get_target_configs()
    feature_validations = [validate_feature_set(config, set(base_df.columns)) for config in configs]
    metrics_df, overfitting_df = fit_and_evaluate_models(base_df, configs)

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    overfitting_df.to_csv(OVERFITTING_CSV_PATH, index=False)
    write_validation_report(merge_summary, split_summary, feature_validations, metrics_df, overfitting_df)

    print(f"Validation report: {VALIDATION_REPORT_PATH}")
    print(f"Metrics CSV: {METRICS_CSV_PATH}")
    print(f"Overfitting CSV: {OVERFITTING_CSV_PATH}")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
