from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost_population_weighted_pipeline import (
    ALL_YEARS,
    TEST_YEAR,
    TRAIN_YEARS,
    XGBOOST_DIR,
    build_population_weighted_base_dataset,
    build_xgb_model,
    evaluate_predictions,
    validate_dependencies_and_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

METRICS_PATH = XGBOOST_DIR / "table3_medium_horizon_metrics.csv"
MONTHLY_PATH = XGBOOST_DIR / "table3_medium_horizon_monthly_validation.csv"
LEAKAGE_AUDIT_PATH = XGBOOST_DIR / "table3_medium_horizon_leakage_audit.txt"
FEATURE_IMPORTANCE_PATH = XGBOOST_DIR / "table3_medium_horizon_feature_importance.csv"
INTERPRETATION_PATH = XGBOOST_DIR / "table3_medium_horizon_interpretation.txt"
PREDICTIONS_PATH = XGBOOST_DIR / "table3_medium_horizon_predictions_2025.csv"

ACTUAL_VS_FORECAST_PATH = FIGURES_DIR / "actual_vs_forecast_2025.png"
MONTHLY_RMSE_PLOT_PATH = FIGURES_DIR / "monthly_rmse_plot.png"
RESIDUAL_PLOT_PATH = FIGURES_DIR / "residual_plot.png"
FEATURE_IMPORTANCE_PLOT_PATH = FIGURES_DIR / "feature_importance_plot.png"

MODEL_LABEL = "Table 3 Medium-Horizon Non-Recursive Population-Weighted"
TARGET_COLUMN = "consumption"
MODEL_KEY_FOR_PARAMS = "table3_sarima_guided_population_weighted"

BASE_FEATURES = [
    "weighted_HDD",
    "weighted_CDD",
    "night",
    "weekend",
    "hour",
    "day_of_week",
    "month",
    "is_public_holiday",
    "is_holiday_window",
    "is_bridge_day",
]

CYCLIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

HISTORICAL_PROFILE_FEATURES = [
    "hist_mean_by_hour",
    "hist_mean_by_hour_day_of_week",
    "hist_mean_by_hour_month",
    "prev_month_hour_mean",
    "prev_month_hour_day_of_week_mean",
    "prev_7d_same_hour_mean",
    "prev_30d_same_hour_mean",
    "prev_4week_same_hour_day_of_week_mean",
]

FEATURE_COLUMNS = BASE_FEATURES + CYCLIC_FEATURES + HISTORICAL_PROFILE_FEATURES
FORBIDDEN_FEATURE_PATTERNS = [
    "log_consumption_lag_1h",
    "consumption_lag_1h",
    "log_consumption_seasonal_diff_24h",
]


def group_expanding_mean(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.Series:
    group_sum_prior = df.groupby(group_cols)[value_col].cumsum() - df[value_col]
    group_count_prior = df.groupby(group_cols).cumcount()
    denominator = group_count_prior.replace(0, np.nan)
    return group_sum_prior / denominator


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_year"] = out["datetime"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (out["day_of_year"] - 1) / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (out["day_of_year"] - 1) / 365.25)
    return out


def add_historical_profile_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy().sort_values("datetime").reset_index(drop=True)
    df = add_cyclic_features(df)
    df["year_month"] = df["datetime"].dt.to_period("M")
    df["prev_month_period"] = df["year_month"] - 1

    df["hist_mean_by_hour"] = group_expanding_mean(df, ["hour"], TARGET_COLUMN)
    df["hist_mean_by_hour_day_of_week"] = group_expanding_mean(df, ["hour", "day_of_week"], TARGET_COLUMN)
    df["hist_mean_by_hour_month"] = group_expanding_mean(df, ["hour", "month"], TARGET_COLUMN)

    month_hour_profile = (
        df.groupby(["year_month", "hour"], as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "prev_month_hour_mean"})
    )
    month_hour_profile["target_year_month"] = month_hour_profile["year_month"] + 1
    df = df.merge(
        month_hour_profile[["target_year_month", "hour", "prev_month_hour_mean"]].rename(
            columns={"target_year_month": "year_month"}
        ),
        on=["year_month", "hour"],
        how="left",
        validate="many_to_one",
    )

    month_hour_dow_profile = (
        df.groupby(["year_month", "hour", "day_of_week"], as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "prev_month_hour_day_of_week_mean"})
    )
    month_hour_dow_profile["target_year_month"] = month_hour_dow_profile["year_month"] + 1
    df = df.merge(
        month_hour_dow_profile[
            ["target_year_month", "hour", "day_of_week", "prev_month_hour_day_of_week_mean"]
        ].rename(columns={"target_year_month": "year_month"}),
        on=["year_month", "hour", "day_of_week"],
        how="left",
        validate="many_to_one",
    )

    df["prev_7d_same_hour_mean"] = (
        df.groupby("hour")[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=7).mean())
    )
    df["prev_30d_same_hour_mean"] = (
        df.groupby("hour")[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(window=30, min_periods=14).mean())
    )
    df["prev_4week_same_hour_day_of_week_mean"] = (
        df.groupby(["hour", "day_of_week"])[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).rolling(window=4, min_periods=4).mean())
    )
    return df


def build_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    augmented_df = add_historical_profile_features(base_df)
    selected_columns = ["datetime", "year", TARGET_COLUMN, *FEATURE_COLUMNS]
    dataset = augmented_df[selected_columns].copy()
    numeric_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    return dataset


def fit_and_predict(dataset: pd.DataFrame) -> dict[str, object]:
    train_df = dataset.loc[dataset["year"].isin(TRAIN_YEARS)].copy()
    test_df = dataset.loc[dataset["year"] == TEST_YEAR].copy()

    model = build_xgb_model(MODEL_KEY_FOR_PARAMS)
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    train_pred = model.predict(train_df[FEATURE_COLUMNS])
    test_pred = model.predict(test_df[FEATURE_COLUMNS])
    train_metrics = evaluate_predictions(train_df[TARGET_COLUMN], train_pred)
    test_metrics = evaluate_predictions(test_df[TARGET_COLUMN], test_pred)

    test_output = test_df[["datetime", TARGET_COLUMN]].copy()
    test_output = test_output.rename(columns={TARGET_COLUMN: "actual_consumption"})
    test_output["predicted_consumption"] = test_pred
    test_output["residual"] = test_output["actual_consumption"] - test_output["predicted_consumption"]

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )
    family_map = {}
    for feature in FEATURE_COLUMNS:
        if feature in {"weighted_HDD", "weighted_CDD"}:
            family_map[feature] = "weather"
        elif feature in BASE_FEATURES:
            family_map[feature] = "calendar"
        elif feature in CYCLIC_FEATURES:
            family_map[feature] = "cyclic"
        else:
            family_map[feature] = "historical_profile"
    feature_importance["family"] = feature_importance["feature"].map(family_map)
    feature_importance = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "train_df": train_df,
        "test_df": test_df,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_output": test_output,
        "feature_importance": feature_importance,
    }


def load_comparator_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    original_train = json.loads((XGBOOST_DIR / "table3_sarima_guided_population_weighted_train_metrics.json").read_text(encoding="utf-8"))
    original_test = json.loads((XGBOOST_DIR / "table3_sarima_guided_population_weighted_metrics.json").read_text(encoding="utf-8"))
    rows.append(
        {
            "model_name": "A) Original Table 3 One-Step-Ahead",
            "train_R2": original_train["train_metrics_consumption_scale"]["R2"],
            "train_MAE": original_train["train_metrics_consumption_scale"]["MAE"],
            "train_RMSE": original_train["train_metrics_consumption_scale"]["RMSE"],
            "train_MAPE": original_train["train_metrics_consumption_scale"]["MAPE"],
            "test_R2": original_test["test_metrics_consumption_scale"]["R2"],
            "test_MAE": original_test["test_metrics_consumption_scale"]["MAE"],
            "test_RMSE": original_test["test_metrics_consumption_scale"]["RMSE"],
            "test_MAPE": original_test["test_metrics_consumption_scale"]["MAPE"],
            "gap_R2": original_train["train_metrics_consumption_scale"]["R2"] - original_test["test_metrics_consumption_scale"]["R2"],
            "gap_RMSE": original_test["test_metrics_consumption_scale"]["RMSE"] - original_train["train_metrics_consumption_scale"]["RMSE"],
            "gap_MAPE": original_test["test_metrics_consumption_scale"]["MAPE"] - original_train["train_metrics_consumption_scale"]["MAPE"],
            "target": "consumption",
            "workflow": "one_step_ahead_actual_lag",
        }
    )

    rolling_metrics = pd.read_csv(XGBOOST_DIR / "table3_2025_rolling_backtest_metrics.csv")
    rolling_row = rolling_metrics.loc[rolling_metrics["model_name"] == "Table 3 Rolling Month-Ahead"].iloc[0]
    rows.append(
        {
            "model_name": "B) Current Table 3 Rolling Month-Ahead",
            "train_R2": float(rolling_row["train_R2"]),
            "train_MAE": float(rolling_row["train_MAE"]),
            "train_RMSE": float(rolling_row["train_RMSE"]),
            "train_MAPE": float(rolling_row["train_MAPE"]),
            "test_R2": float(rolling_row["test_R2"]),
            "test_MAE": float(rolling_row["test_MAE"]),
            "test_RMSE": float(rolling_row["test_RMSE"]),
            "test_MAPE": float(rolling_row["test_MAPE"]),
            "gap_R2": float(rolling_row["gap_R2"]),
            "gap_RMSE": float(rolling_row["gap_RMSE"]),
            "gap_MAPE": float(rolling_row["gap_MAPE"]),
            "target": "consumption",
            "workflow": "monthly_refit_recursive_within_month",
        }
    )

    table8_payload = json.loads(
        (XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_metrics.json").read_text(encoding="utf-8")
    )
    rows.append(
        {
            "model_name": "C) Table 8 Non-Recursive Historical",
            "train_R2": table8_payload["train_metrics"]["R2"],
            "train_MAE": table8_payload["train_metrics"]["MAE"],
            "train_RMSE": table8_payload["train_metrics"]["RMSE"],
            "train_MAPE": table8_payload["train_metrics"]["MAPE"],
            "test_R2": table8_payload["test_metrics"]["R2"],
            "test_MAE": table8_payload["test_metrics"]["MAE"],
            "test_RMSE": table8_payload["test_metrics"]["RMSE"],
            "test_MAPE": table8_payload["test_metrics"]["MAPE"],
            "gap_R2": table8_payload["train_test_gap"]["R2_train_minus_test"],
            "gap_RMSE": table8_payload["train_test_gap"]["RMSE_test_minus_train"],
            "gap_MAPE": table8_payload["train_test_gap"]["MAPE_test_minus_train"],
            "target": "sqrt_consumption",
            "workflow": "non_recursive_historical_analog",
        }
    )
    return pd.DataFrame(rows)


def compute_monthly_metrics(actual: pd.Series, pred: pd.Series, datetimes: pd.Series) -> pd.DataFrame:
    temp = pd.DataFrame({"datetime": pd.to_datetime(datetimes), "actual": actual, "pred": pred})
    rows = []
    for month, month_df in temp.groupby(temp["datetime"].dt.month):
        metrics = evaluate_predictions(month_df["actual"], month_df["pred"])
        rows.append(
            {
                "month": int(month),
                "R2": float(metrics["R2"]),
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "MAPE": float(metrics["MAPE"]),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_monthly_validation(test_output: pd.DataFrame) -> pd.DataFrame:
    medium = compute_monthly_metrics(
        test_output["actual_consumption"], test_output["predicted_consumption"], test_output["datetime"]
    ).rename(
        columns={
            "R2": "medium_horizon_R2",
            "MAE": "medium_horizon_MAE",
            "RMSE": "medium_horizon_RMSE",
            "MAPE": "medium_horizon_MAPE",
        }
    )

    original = pd.read_csv(XGBOOST_DIR / "table3_sarima_guided_population_weighted_predictions_2025.csv", parse_dates=["datetime"])
    original_monthly = compute_monthly_metrics(
        original["consumption"], original["prediction_consumption"], original["datetime"]
    ).rename(
        columns={
            "R2": "original_one_step_R2",
            "MAE": "original_one_step_MAE",
            "RMSE": "original_one_step_RMSE",
            "MAPE": "original_one_step_MAPE",
        }
    )

    rolling = pd.read_csv(XGBOOST_DIR / "table3_2025_rolling_vs_recursive_comparison.csv", parse_dates=["datetime"])
    rolling_monthly = compute_monthly_metrics(
        rolling["actual_consumption"], rolling["rolling_month_ahead_forecast"], rolling["datetime"]
    ).rename(
        columns={
            "R2": "rolling_month_ahead_R2",
            "MAE": "rolling_month_ahead_MAE",
            "RMSE": "rolling_month_ahead_RMSE",
            "MAPE": "rolling_month_ahead_MAPE",
        }
    )

    table8 = pd.read_csv(XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_predictions_2025.csv", parse_dates=["datetime"])
    table8_monthly = compute_monthly_metrics(
        table8["actual_consumption"], table8["predicted_consumption"], table8["datetime"]
    ).rename(
        columns={
            "R2": "table8_R2",
            "MAE": "table8_MAE",
            "RMSE": "table8_RMSE",
            "MAPE": "table8_MAPE",
        }
    )

    monthly = medium.merge(original_monthly, on="month").merge(rolling_monthly, on="month").merge(table8_monthly, on="month")
    monthly["delta_vs_original_R2"] = monthly["medium_horizon_R2"] - monthly["original_one_step_R2"]
    monthly["delta_vs_original_RMSE"] = monthly["medium_horizon_RMSE"] - monthly["original_one_step_RMSE"]
    monthly["delta_vs_rolling_R2"] = monthly["medium_horizon_R2"] - monthly["rolling_month_ahead_R2"]
    monthly["delta_vs_rolling_RMSE"] = monthly["medium_horizon_RMSE"] - monthly["rolling_month_ahead_RMSE"]
    monthly["delta_vs_table8_R2"] = monthly["medium_horizon_R2"] - monthly["table8_R2"]
    monthly["delta_vs_table8_RMSE"] = monthly["medium_horizon_RMSE"] - monthly["table8_RMSE"]
    monthly["rmse_rank_desc"] = monthly["medium_horizon_RMSE"].rank(method="dense", ascending=False).astype(int)
    monthly["r2_rank_asc"] = monthly["medium_horizon_R2"].rank(method="dense", ascending=True).astype(int)
    monthly["is_weak_month"] = ((monthly["rmse_rank_desc"] <= 3) | (monthly["r2_rank_asc"] <= 3)).astype(int)
    monthly["is_strong_month"] = (
        (monthly["medium_horizon_RMSE"].rank(method="dense", ascending=True) <= 3)
        | (monthly["medium_horizon_R2"].rank(method="dense", ascending=False) <= 3)
    ).astype(int)
    return monthly


def run_leakage_audit(base_df: pd.DataFrame) -> str:
    lines = [
        "Table 3 Medium-Horizon Leakage Audit",
        "",
        "Forbidden direct-lag features:",
        f"- Forbidden patterns checked: {FORBIDDEN_FEATURE_PATTERNS}",
        f"- Present in final feature set: {[feature for feature in FEATURE_COLUMNS if feature in FORBIDDEN_FEATURE_PATTERNS]}",
        "",
        "Feature family rules:",
        "- No log_consumption_lag_1h.",
        "- No consumption_lag_1h.",
        "- No log_consumption_seasonal_diff_24h.",
        "- No recursive predicted feedback: features are constructed only from observed history.",
        "",
        "Empirical forward-leakage audit:",
    ]

    original = add_historical_profile_features(base_df)
    cutoffs = [
        pd.Timestamp("2025-01-31 23:00:00"),
        pd.Timestamp("2025-03-15 12:00:00"),
        pd.Timestamp("2025-06-30 23:00:00"),
        pd.Timestamp("2025-09-30 23:00:00"),
    ]

    for cutoff in cutoffs:
        perturbed = base_df.copy()
        future_mask = perturbed["datetime"] > cutoff
        perturbed.loc[future_mask, TARGET_COLUMN] = perturbed.loc[future_mask, TARGET_COLUMN] + 99999.0
        recomputed = add_historical_profile_features(perturbed)
        compare_mask = original["datetime"] <= cutoff
        max_abs_diff = 0.0
        changed_features: list[str] = []
        for feature in HISTORICAL_PROFILE_FEATURES:
            orig_values = original.loc[compare_mask, feature].to_numpy()
            new_values = recomputed.loc[compare_mask, feature].to_numpy()
            diff = np.nanmax(np.abs(orig_values - new_values))
            if np.isfinite(diff) and diff > max_abs_diff:
                max_abs_diff = float(diff)
            if np.isfinite(diff) and diff > 1e-9:
                changed_features.append(feature)
        lines.append(
            f"- Cutoff {cutoff}: max absolute feature difference for rows <= cutoff = {max_abs_diff:.12f}; changed_features={changed_features if changed_features else 'None'}"
        )

    lines.extend(
        [
            "",
            "Feature-generation timeline summary:",
            "- hist_mean_by_hour / hist_mean_by_hour_day_of_week / hist_mean_by_hour_month use strict prior grouped sums and counts only.",
            "- prev_month_hour_mean / prev_month_hour_day_of_week_mean use completed previous-month profiles only.",
            "- prev_7d_same_hour_mean uses the previous seven same-hour observations via shift(1) within each hour group.",
            "- prev_30d_same_hour_mean uses the previous same-hour profile window only.",
            "- prev_4week_same_hour_day_of_week_mean uses only prior weekly occurrences for the same hour/day-of-week combination.",
            "",
            "Leakage-safe conclusion:",
            "LEAKAGE_SAFE = True, provided the model is interpreted as a non-recursive historical-profile design that may use actual completed history strictly before each target timestamp.",
        ]
    )
    return "\n".join(lines)


def plot_outputs(test_output: pd.DataFrame, monthly_validation: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    original = pd.read_csv(XGBOOST_DIR / "table3_sarima_guided_population_weighted_predictions_2025.csv", parse_dates=["datetime"])
    rolling = pd.read_csv(XGBOOST_DIR / "table3_2025_rolling_vs_recursive_comparison.csv", parse_dates=["datetime"])
    table8 = pd.read_csv(XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_predictions_2025.csv", parse_dates=["datetime"])

    compare_plot = test_output[["datetime", "actual_consumption", "predicted_consumption"]].rename(
        columns={"predicted_consumption": "medium_horizon_forecast"}
    )
    compare_plot = compare_plot.merge(
        original[["datetime", "prediction_consumption"]].rename(columns={"prediction_consumption": "original_one_step_forecast"}),
        on="datetime",
        how="left",
    ).merge(
        rolling[["datetime", "rolling_month_ahead_forecast"]],
        on="datetime",
        how="left",
    ).merge(
        table8[["datetime", "predicted_consumption"]].rename(columns={"predicted_consumption": "table8_forecast"}),
        on="datetime",
        how="left",
    )
    compare_plot["date"] = compare_plot["datetime"].dt.normalize()
    daily = (
        compare_plot.groupby("date", as_index=False)[
            [
                "actual_consumption",
                "medium_horizon_forecast",
                "original_one_step_forecast",
                "rolling_month_ahead_forecast",
                "table8_forecast",
            ]
        ]
        .mean()
    )

    plt.figure(figsize=(16, 6))
    plt.plot(daily["date"], daily["actual_consumption"], label="Actual 2025", linewidth=1.4)
    plt.plot(daily["date"], daily["medium_horizon_forecast"], label="Medium-Horizon", linewidth=1.1)
    plt.plot(daily["date"], daily["original_one_step_forecast"], label="Original Table 3 One-Step", linewidth=0.9)
    plt.plot(daily["date"], daily["rolling_month_ahead_forecast"], label="Rolling Month-Ahead", linewidth=0.9)
    plt.plot(daily["date"], daily["table8_forecast"], label="Table 8 Final", linewidth=0.9)
    plt.title("2025 Actual vs Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_VS_FORECAST_PATH, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_validation["month"], monthly_validation["medium_horizon_RMSE"], marker="o", label="Medium-Horizon")
    plt.plot(monthly_validation["month"], monthly_validation["original_one_step_RMSE"], marker="o", label="Original One-Step")
    plt.plot(monthly_validation["month"], monthly_validation["rolling_month_ahead_RMSE"], marker="o", label="Rolling Month-Ahead")
    plt.plot(monthly_validation["month"], monthly_validation["table8_RMSE"], marker="o", label="Table 8 Final")
    plt.title("Monthly RMSE Comparison")
    plt.xlabel("Month")
    plt.ylabel("RMSE")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(MONTHLY_RMSE_PLOT_PATH, dpi=160)
    plt.close()

    residual_daily = test_output.copy()
    residual_daily["date"] = residual_daily["datetime"].dt.normalize()
    residual_daily = residual_daily.groupby("date", as_index=False)["residual"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(residual_daily["date"], residual_daily["residual"], linewidth=1.0)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Medium-Horizon Residuals Over Time")
    axes[0].set_ylabel("Daily Mean Residual")

    axes[1].hist(test_output["residual"], bins=60, alpha=0.8)
    axes[1].set_title("Medium-Horizon Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH, dpi=160)
    plt.close()

    top_features = feature_importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Feature Importance - Medium-Horizon Table 3 Variant")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    validate_dependencies_and_files()
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base_df, _ = build_population_weighted_base_dataset(ALL_YEARS)
    dataset = build_dataset(base_df)
    result = fit_and_predict(dataset)
    result["test_output"].to_csv(PREDICTIONS_PATH, index=False)
    result["feature_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    comparators = load_comparator_metrics()
    medium_row = pd.DataFrame(
        [
            {
                "model_name": "D) Table 3 Medium-Horizon Non-Recursive",
                "train_R2": result["train_metrics"]["R2"],
                "train_MAE": result["train_metrics"]["MAE"],
                "train_RMSE": result["train_metrics"]["RMSE"],
                "train_MAPE": result["train_metrics"]["MAPE"],
                "test_R2": result["test_metrics"]["R2"],
                "test_MAE": result["test_metrics"]["MAE"],
                "test_RMSE": result["test_metrics"]["RMSE"],
                "test_MAPE": result["test_metrics"]["MAPE"],
                "gap_R2": result["train_metrics"]["R2"] - result["test_metrics"]["R2"],
                "gap_RMSE": result["test_metrics"]["RMSE"] - result["train_metrics"]["RMSE"],
                "gap_MAPE": result["test_metrics"]["MAPE"] - result["train_metrics"]["MAPE"],
                "target": TARGET_COLUMN,
                "workflow": "non_recursive_historical_profiles_no_direct_hourly_lag",
            }
        ]
    )
    metrics_df = pd.concat([comparators, medium_row], ignore_index=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

    monthly_validation = build_monthly_validation(result["test_output"])
    monthly_validation.to_csv(MONTHLY_PATH, index=False)

    leakage_text = run_leakage_audit(base_df)
    LEAKAGE_AUDIT_PATH.write_text(leakage_text, encoding="utf-8")

    plot_outputs(result["test_output"], monthly_validation, result["feature_importance"])

    family_importance = (
        result["feature_importance"].groupby("family", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )
    weak_months = monthly_validation.loc[monthly_validation["is_weak_month"] == 1, ["month", "medium_horizon_RMSE", "medium_horizon_MAPE", "medium_horizon_R2"]]
    strong_months = monthly_validation.loc[monthly_validation["is_strong_month"] == 1, ["month", "medium_horizon_RMSE", "medium_horizon_MAPE", "medium_horizon_R2"]]

    medium_metrics = metrics_df.loc[metrics_df["model_name"] == "D) Table 3 Medium-Horizon Non-Recursive"].iloc[0]
    original_metrics = metrics_df.loc[metrics_df["model_name"] == "A) Original Table 3 One-Step-Ahead"].iloc[0]
    rolling_metrics = metrics_df.loc[metrics_df["model_name"] == "B) Current Table 3 Rolling Month-Ahead"].iloc[0]
    table8_metrics = metrics_df.loc[metrics_df["model_name"] == "C) Table 8 Non-Recursive Historical"].iloc[0]

    interpretation_lines = [
        "Table 3 Medium-Horizon Non-Recursive Variant",
        "",
        "Core question:",
        "Can we retain SARIMA-style temporal intelligence WITHOUT direct hourly actual lag dependence?",
        "",
        "Answer framework:",
        "- This model removes direct hourly lag usage and recursive predicted feedback.",
        "- It keeps temporal intelligence through calendar seasonality, cyclic encodings, and leakage-safe historical profile features built from completed history only.",
        "",
        "Overall metrics comparison:",
        metrics_df.to_string(index=False),
        "",
        "Medium-horizon monthly stability:",
        monthly_validation.to_string(index=False),
        "",
        "Feature family importance:",
        family_importance.to_string(index=False),
        "",
        "Top individual features:",
        result["feature_importance"].head(15).to_string(index=False),
        "",
        "Weak months:",
        weak_months.to_string(index=False),
        "",
        "Strong months:",
        strong_months.to_string(index=False),
        "",
        "Interpretation:",
        (
            f"- Relative to the original one-step model, the medium-horizon variant changes test R2 by "
            f"{medium_metrics['test_R2'] - original_metrics['test_R2']:.4f}, RMSE by "
            f"{medium_metrics['test_RMSE'] - original_metrics['test_RMSE']:.2f}, and MAPE by "
            f"{medium_metrics['test_MAPE'] - original_metrics['test_MAPE']:.2f}."
        ),
        (
            f"- Relative to the current rolling month-ahead workflow, the medium-horizon variant changes test R2 by "
            f"{medium_metrics['test_R2'] - rolling_metrics['test_R2']:.4f}, RMSE by "
            f"{medium_metrics['test_RMSE'] - rolling_metrics['test_RMSE']:.2f}, and MAPE by "
            f"{medium_metrics['test_MAPE'] - rolling_metrics['test_MAPE']:.2f}."
        ),
        (
            f"- Relative to the final Table 8 model, the medium-horizon variant changes test R2 by "
            f"{medium_metrics['test_R2'] - table8_metrics['test_R2']:.4f}, RMSE by "
            f"{medium_metrics['test_RMSE'] - table8_metrics['test_RMSE']:.2f}, and MAPE by "
            f"{medium_metrics['test_MAPE'] - table8_metrics['test_MAPE']:.2f}."
        ),
        (
            "- If historical_profile family dominates feature importance and the model clearly outperforms the operational recursive variants, "
            "that indicates we can preserve a substantial amount of SARIMA-style temporal intelligence without direct hourly actual lags."
        ),
        (
            "- If weather and calendar features remain materially important, then the model is not merely replaying seasonal averages; "
            "it is blending temporal memory with exogenous temperature structure."
        ),
        "",
        "Leakage audit conclusion:",
        "Refer to table3_medium_horizon_leakage_audit.txt. The intended interpretation is non-recursive, historical-profile-based, and direct-lag-free.",
    ]
    INTERPRETATION_PATH.write_text("\n".join(interpretation_lines), encoding="utf-8")

    print("Table 3 medium-horizon experiment complete.")
    print(
        f"Medium-horizon test R2/RMSE/MAPE: {result['test_metrics']['R2']:.4f} / {result['test_metrics']['RMSE']:.2f} / {result['test_metrics']['MAPE']:.2f}"
    )
    print(
        "Vs rolling month-ahead: "
        f"dR2={medium_metrics['test_R2'] - rolling_metrics['test_R2']:.4f}, "
        f"dRMSE={medium_metrics['test_RMSE'] - rolling_metrics['test_RMSE']:.2f}, "
        f"dMAPE={medium_metrics['test_MAPE'] - rolling_metrics['test_MAPE']:.2f}"
    )
    print(f"Saved: {METRICS_PATH}")
    print(f"Saved: {MONTHLY_PATH}")
    print(f"Saved: {LEAKAGE_AUDIT_PATH}")
    print(f"Saved: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved: {INTERPRETATION_PATH}")


if __name__ == "__main__":
    main()
