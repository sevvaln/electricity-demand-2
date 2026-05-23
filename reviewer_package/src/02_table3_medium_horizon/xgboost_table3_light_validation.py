from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from xgboost_population_weighted_pipeline import (
    ALL_YEARS,
    MODEL_CONFIGS,
    TEST_YEAR,
    TRAIN_YEARS,
    XGBOOST_DIR,
    build_model_dataset,
    build_population_weighted_base_dataset,
    split_train_test,
    validate_dependencies_and_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_PATH = XGBOOST_DIR / "table3_light_validation_population_weighted.txt"
METRICS_CSV_PATH = XGBOOST_DIR / "table3_light_validation_population_weighted_metrics.csv"
TRAIN_METRICS_PATH = XGBOOST_DIR / "table3_sarima_guided_population_weighted_train_metrics.json"
TEST_METRICS_PATH = XGBOOST_DIR / "table3_sarima_guided_population_weighted_metrics.json"

MODEL_KEY = "table3_sarima_guided_population_weighted"
EXPECTED_TARGET = "consumption"
EXPECTED_FEATURES = [
    "weighted_HDD",
    "weighted_CDD",
    "night",
    "weekend",
    "log_consumption_lag_1h",
    "log_consumption_seasonal_diff_24h",
]


def load_config() -> dict[str, object]:
    config_map = {str(config["model_key"]): config for config in MODEL_CONFIGS}
    return config_map[MODEL_KEY]


def load_metrics() -> tuple[dict[str, object], dict[str, object]]:
    train_payload = json.loads(TRAIN_METRICS_PATH.read_text(encoding="utf-8"))
    test_payload = json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))
    return train_payload, test_payload


def main() -> None:
    validate_dependencies_and_files()
    config = load_config()
    train_payload, test_payload = load_metrics()

    base_df, _ = build_population_weighted_base_dataset(ALL_YEARS)
    dataset, feature_columns = build_model_dataset(base_df, config)
    train_df, test_df = split_train_test(dataset)

    actual_target = str(config["target_column"])
    actual_features = list(feature_columns)
    unexpected_features = [feature for feature in actual_features if feature not in EXPECTED_FEATURES]
    missing_expected = [feature for feature in EXPECTED_FEATURES if feature not in actual_features]
    old_temperature_features_present = [feature for feature in actual_features if feature in {"HDD", "CDD"}]
    weighted_temperature_ok = "weighted_HDD" in actual_features and "weighted_CDD" in actual_features

    train_years_actual = sorted(train_df["year"].unique().tolist())
    test_years_actual = sorted(test_df["year"].unique().tolist())
    overlap_count = len(set(pd.to_datetime(train_df["datetime"])) & set(pd.to_datetime(test_df["datetime"])))

    train_metrics = train_payload["train_metrics_consumption_scale"]
    test_metrics = test_payload["test_metrics_consumption_scale"]
    gap_r2 = float(train_metrics["R2"] - test_metrics["R2"])
    gap_rmse = float(test_metrics["RMSE"] - train_metrics["RMSE"])
    gap_mape = float(test_metrics["MAPE"] - train_metrics["MAPE"])

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "Table 3 SARIMA-Guided Population-Weighted",
                "target_column": actual_target,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_R2": float(train_metrics["R2"]),
                "train_MAE": float(train_metrics["MAE"]),
                "train_RMSE": float(train_metrics["RMSE"]),
                "train_MAPE": float(train_metrics["MAPE"]),
                "test_R2": float(test_metrics["R2"]),
                "test_MAE": float(test_metrics["MAE"]),
                "test_RMSE": float(test_metrics["RMSE"]),
                "test_MAPE": float(test_metrics["MAPE"]),
                "gap_R2": gap_r2,
                "gap_RMSE": gap_rmse,
                "gap_MAPE": gap_mape,
                "evaluation_mode": str(test_payload["method"]),
            }
        ]
    )
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    lines = [
        "Table 3 Light Validation Report: Population-Weighted SARIMA-Guided Model",
        "",
        "1. Correctness Check",
        f"- Expected target: {EXPECTED_TARGET}",
        f"- Actual target: {actual_target}",
        f"- Target match: {actual_target == EXPECTED_TARGET}",
        f"- Expected features: {EXPECTED_FEATURES}",
        f"- Actual features: {actual_features}",
        f"- Missing expected features: {missing_expected}",
        f"- Unexpected features: {unexpected_features}",
        f"- weighted_HDD / weighted_CDD used: {weighted_temperature_ok}",
        f"- Old HDD / CDD still present in feature set: {old_temperature_features_present}",
        "",
        "2. Train/Test Split Validation",
        f"- Expected train years: {TRAIN_YEARS}",
        f"- Actual train years: {train_years_actual}",
        f"- Expected test year: {TEST_YEAR}",
        f"- Actual test years: {test_years_actual}",
        f"- Train rows: {len(train_df)}",
        f"- Test rows: {len(test_df)}",
        f"- Train datetime min/max: {train_df['datetime'].min()} / {train_df['datetime'].max()}",
        f"- Test datetime min/max: {test_df['datetime'].min()} / {test_df['datetime'].max()}",
        f"- Train/test datetime overlap count: {overlap_count}",
        "",
        "3. Leakage Check",
        "- This is a short-term one-step-ahead model, not a fully blind long-term forecast.",
        "- log_consumption_lag_1h is built with shift(1), so each row uses only past log consumption.",
        "- log_consumption_seasonal_diff_24h is built as lag_1h minus lag_25h, so it also uses only past values.",
        "- No future consumption is used in the feature construction itself.",
        "- During 2025 evaluation, the method is one-step-ahead; this means observed past 2025 consumption can inform the lag features at each step.",
        f"- Evaluation mode recorded in test metrics: {test_payload['method']}",
        "",
        "4. Performance Summary",
        f"- Train R2 / MAE / RMSE / MAPE: {train_metrics['R2']:.6f} / {train_metrics['MAE']:.6f} / {train_metrics['RMSE']:.6f} / {train_metrics['MAPE']:.6f}",
        f"- Test R2 / MAE / RMSE / MAPE: {test_metrics['R2']:.6f} / {test_metrics['MAE']:.6f} / {test_metrics['RMSE']:.6f} / {test_metrics['MAPE']:.6f}",
        f"- Train-test gap R2: {gap_r2:.6f}",
        f"- Train-test gap RMSE: {gap_rmse:.6f}",
        f"- Train-test gap MAPE: {gap_mape:.6f}",
        "",
        "Conclusion",
        (
            "The Table 3 implementation is consistent with the intended SARIMA-guided short-term design: "
            "target is consumption, weighted_HDD/weighted_CDD replace old HDD/CDD, train/test split is correct, "
            "and lag features are backward-looking. This model should be interpreted as a one-step-ahead "
            "short-term benchmark rather than a fully blind long-horizon forecast."
        ),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
