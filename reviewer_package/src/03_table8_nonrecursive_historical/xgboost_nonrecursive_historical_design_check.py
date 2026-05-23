from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from xgboost_population_weighted_nonrecursive_historical_experiment import (
    BASELINE_FEATURES,
)
from xgboost_population_weighted_pipeline import (
    TEST_YEAR,
    TRAIN_YEARS,
    XGBOOST_DIR,
    build_population_weighted_base_dataset,
    evaluate_predictions,
    invert_predictions,
    split_train_test,
    validate_dependencies_and_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_METRICS_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_metrics.json"
OUTPUT_CSV_PATH = XGBOOST_DIR / "table8_historical_profile_design_check_2025.csv"
OUTPUT_TXT_PATH = XGBOOST_DIR / "table8_historical_profile_design_check_2025.txt"

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

EXACT_PREV_YEAR_ANALOG_FEATURES = [
    "prev_year_same_month_day_hour_sqrt",
    "prev_year_same_month_day_prev_hour_sqrt",
    "prev_year_same_month_day_seasonal_diff_24h_sqrt",
]

GROUPED_PROFILE_FEATURES = [
    "prev_year_mean_sqrt_by_hour_day_of_week",
    "prev_year_mean_sqrt_by_hour_month",
]

ALL_FEATURES = BASELINE_FEATURES + CYCLIC_FEATURES + EXACT_PREV_YEAR_ANALOG_FEATURES + GROUPED_PROFILE_FEATURES


def load_final_hyperparameters() -> dict[str, object]:
    payload = json.loads(FINAL_METRICS_PATH.read_text(encoding="utf-8"))
    return dict(payload["hyperparameters"])


def add_common_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("datetime").reset_index(drop=True)
    out["day"] = out["datetime"].dt.day
    out["day_of_year"] = out["datetime"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * (out["day_of_year"] - 1) / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * (out["day_of_year"] - 1) / 365.25)
    out["sqrt_consumption_seasonal_diff_24h"] = out["sqrt_consumption"] - out["sqrt_consumption"].shift(24)
    return out


def add_historical_features(base_df: pd.DataFrame, grouped_mode: str) -> pd.DataFrame:
    df = add_common_calendar_features(base_df)

    for feature in EXACT_PREV_YEAR_ANALOG_FEATURES + GROUPED_PROFILE_FEATURES:
        df[feature] = np.nan

    available_years = sorted(df["year"].unique())
    for target_year in available_years:
        source_year = target_year - 1
        if source_year not in available_years:
            continue

        exact_source_df = df.loc[df["year"] == source_year].copy()
        target_mask = df["year"] == target_year
        target_df = df.loc[target_mask, ["month", "day", "hour", "day_of_week"]].copy()

        direct_lookup = (
            exact_source_df[["month", "day", "hour", "sqrt_consumption"]]
            .rename(columns={"sqrt_consumption": "prev_year_same_month_day_hour_sqrt"})
        )
        target_df = target_df.merge(direct_lookup, on=["month", "day", "hour"], how="left")

        prev_hour_lookup = exact_source_df[["datetime", "sqrt_consumption"]].copy()
        prev_hour_lookup["mapped_time"] = prev_hour_lookup["datetime"] + pd.Timedelta(hours=1)
        prev_hour_lookup["month"] = prev_hour_lookup["mapped_time"].dt.month
        prev_hour_lookup["day"] = prev_hour_lookup["mapped_time"].dt.day
        prev_hour_lookup["hour"] = prev_hour_lookup["mapped_time"].dt.hour
        prev_hour_lookup = prev_hour_lookup[
            ["month", "day", "hour", "sqrt_consumption"]
        ].rename(columns={"sqrt_consumption": "prev_year_same_month_day_prev_hour_sqrt"})
        target_df = target_df.merge(prev_hour_lookup, on=["month", "day", "hour"], how="left")

        seasonal_lookup = (
            exact_source_df[["month", "day", "hour", "sqrt_consumption_seasonal_diff_24h"]]
            .rename(
                columns={
                    "sqrt_consumption_seasonal_diff_24h": "prev_year_same_month_day_seasonal_diff_24h_sqrt"
                }
            )
        )
        target_df = target_df.merge(seasonal_lookup, on=["month", "day", "hour"], how="left")

        if grouped_mode == "previous_year_only":
            grouped_source_df = exact_source_df
        elif grouped_mode == "multi_year":
            grouped_source_df = df.loc[df["year"] < target_year].copy()
        else:
            raise ValueError(f"Unknown grouped_mode: {grouped_mode}")

        hour_dow_lookup = (
            grouped_source_df.groupby(["hour", "day_of_week"], as_index=False)["sqrt_consumption"]
            .mean()
            .rename(columns={"sqrt_consumption": "prev_year_mean_sqrt_by_hour_day_of_week"})
        )
        target_df = target_df.merge(hour_dow_lookup, on=["hour", "day_of_week"], how="left")

        hour_month_lookup = (
            grouped_source_df.groupby(["hour", "month"], as_index=False)["sqrt_consumption"]
            .mean()
            .rename(columns={"sqrt_consumption": "prev_year_mean_sqrt_by_hour_month"})
        )
        target_df = target_df.merge(hour_month_lookup, on=["hour", "month"], how="left")

        for feature in EXACT_PREV_YEAR_ANALOG_FEATURES + GROUPED_PROFILE_FEATURES:
            df.loc[target_mask, feature] = target_df[feature].to_numpy()

    return df


def build_dataset(grouped_mode: str) -> pd.DataFrame:
    base_df, _ = build_population_weighted_base_dataset(TRAIN_YEARS + [TEST_YEAR])
    augmented_df = add_historical_features(base_df, grouped_mode=grouped_mode)
    selected_columns = ["datetime", "year", "consumption", "sqrt_consumption", *ALL_FEATURES]
    dataset = augmented_df[selected_columns].copy()
    dataset = dataset.apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name != "datetime"
        else column
    )
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    return dataset


def fit_and_evaluate(grouped_mode: str, params: dict[str, object]) -> dict[str, object]:
    dataset = build_dataset(grouped_mode)
    train_df, test_df = split_train_test(dataset)
    model = XGBRegressor(**params)
    model.fit(train_df[ALL_FEATURES], train_df["sqrt_consumption"])

    train_raw = model.predict(train_df[ALL_FEATURES])
    test_raw = model.predict(test_df[ALL_FEATURES])
    train_pred = invert_predictions("square_to_consumption", train_raw)
    test_pred = invert_predictions("square_to_consumption", test_raw)

    y_train = np.square(train_df["sqrt_consumption"].to_numpy())
    y_test = np.square(test_df["sqrt_consumption"].to_numpy())

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    return {
        "design": grouped_mode,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gap_R2": float(train_metrics["R2"] - test_metrics["R2"]),
        "gap_RMSE": float(test_metrics["RMSE"] - train_metrics["RMSE"]),
        "gap_MAPE": float(test_metrics["MAPE"] - train_metrics["MAPE"]),
        "grouped_profile_source_for_2025": (
            "2024 only" if grouped_mode == "previous_year_only" else "2022-2024 history"
        ),
    }


def main() -> None:
    validate_dependencies_and_files()
    params = load_final_hyperparameters()

    result_a = fit_and_evaluate("previous_year_only", params)
    result_b = fit_and_evaluate("multi_year", params)

    rows = []
    for label, result in [
        ("A_validated_previous_year_only", result_a),
        ("B_multi_year_grouped_profiles", result_b),
    ]:
        rows.append(
            {
                "model_variant": label,
                "grouped_profile_logic": result["design"],
                "grouped_profile_source_for_2025": result["grouped_profile_source_for_2025"],
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "train_R2": float(result["train_metrics"]["R2"]),
                "train_MAE": float(result["train_metrics"]["MAE"]),
                "train_RMSE": float(result["train_metrics"]["RMSE"]),
                "train_MAPE": float(result["train_metrics"]["MAPE"]),
                "test_R2": float(result["test_metrics"]["R2"]),
                "test_MAE": float(result["test_metrics"]["MAE"]),
                "test_RMSE": float(result["test_metrics"]["RMSE"]),
                "test_MAPE": float(result["test_metrics"]["MAPE"]),
                "gap_R2": result["gap_R2"],
                "gap_RMSE": result["gap_RMSE"],
                "gap_MAPE": result["gap_MAPE"],
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)

    diff_r2 = float(result_b["test_metrics"]["R2"] - result_a["test_metrics"]["R2"])
    diff_rmse = float(result_b["test_metrics"]["RMSE"] - result_a["test_metrics"]["RMSE"])
    diff_mape = float(result_b["test_metrics"]["MAPE"] - result_a["test_metrics"]["MAPE"])
    diff_gap_r2 = float(result_b["gap_R2"] - result_a["gap_R2"])
    diff_gap_rmse = float(result_b["gap_RMSE"] - result_a["gap_RMSE"])
    diff_gap_mape = float(result_b["gap_MAPE"] - result_a["gap_MAPE"])

    if diff_r2 > 0.003 and diff_rmse < -40 and diff_gap_rmse <= 10:
        recommendation = (
            "Multi-year grouped historical profiles show a material improvement in 2025 backtest accuracy "
            "without harming generalization materially. Consider adopting OPTION 2 for 2026."
        )
    else:
        recommendation = (
            "The multi-year grouped profile logic does not deliver a clear enough improvement in the controlled "
            "2025 backtest. Preserve the validated previous-year-only specification for 2026 forecasting."
        )

    lines = [
        "Controlled Historical-Feature Design Check for Table 8 Non-Recursive Historical Population-Weighted Forecast",
        "",
        "Goal:",
        "- Compare validated previous-year-only grouped profile logic against a multi-year grouped profile design.",
        "- Keep train/test split, hyperparameters, target, exact previous-year analog features, and evaluation pipeline fixed.",
        "- Change only the grouped historical profile construction logic.",
        "",
        comparison_df.to_string(index=False),
        "",
        "Incremental difference: B minus A",
        f"- test_R2 difference: {diff_r2:.6f}",
        f"- test_RMSE difference: {diff_rmse:.6f}",
        f"- test_MAPE difference: {diff_mape:.6f}",
        f"- gap_R2 difference: {diff_gap_r2:.6f}",
        f"- gap_RMSE difference: {diff_gap_rmse:.6f}",
        f"- gap_MAPE difference: {diff_gap_mape:.6f}",
        "",
        "Recommendation:",
        recommendation,
    ]
    OUTPUT_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Saved: {OUTPUT_CSV_PATH}")
    print(f"Saved: {OUTPUT_TXT_PATH}")


if __name__ == "__main__":
    main()
