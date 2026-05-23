from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from talep_tahmin_tek_dosya import load_economic_data
from xgboost_nonrecursive_historical_model_c_final import FINAL_MODEL_PARAMS
from xgboost_population_weighted_nonrecursive_historical_experiment import EXPERIMENTAL_FEATURES
from xgboost_population_weighted_pipeline import (
    MODELS_DIR,
    WEIGHTED_HDD_CDD_PATH,
    build_population_weighted_base_dataset,
    build_xgb_model,
    build_model_dataset,
    invert_predictions,
    split_train_test,
    validate_dependencies_and_files,
)
from xgboost_train import TURKEY_PUBLIC_HOLIDAYS


PROJECT_ROOT = Path(__file__).resolve().parent
XGBOOST_DIR = PROJECT_ROOT / "outputs" / "xgboost"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FORECAST_INDEX = pd.date_range("2026-01-01 00:00:00", "2026-12-31 23:00:00", freq="h")
TRAIN_YEARS_FINAL = [2022, 2023, 2024, 2025]
FORECAST_YEAR = 2026

TABLE3_FORECAST_PATH = XGBOOST_DIR / "table3_forecast_2026.csv"
TABLE3_NOTE_PATH = XGBOOST_DIR / "table3_forecast_2026_methodology_note.txt"
TABLE3_PREVIOUS_FULL_RECURSIVE_PATH = (
    XGBOOST_DIR / "table3_forecast_2026_full_year_recursive_previous.csv"
)
TABLE3_COMPARISON_PATH = (
    XGBOOST_DIR / "table3_forecast_2026_rolling_vs_full_year_recursive.csv"
)
TABLE3_COMPARISON_NOTE_PATH = (
    XGBOOST_DIR / "table3_forecast_2026_rolling_vs_full_year_recursive.txt"
)
TABLE8_FORECAST_PATH = (
    XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_forecast_2026.csv"
)
TABLE8_NOTE_PATH = XGBOOST_DIR / "table8_forecast_2026_methodology_note.txt"
TABLE8_AUDIT_PATH = XGBOOST_DIR / "table8_forecast_2026_feature_generation_audit.txt"
SUMMARY_PATH = XGBOOST_DIR / "forecast_2026_model_comparison.txt"

FULL_YEAR_PLOT_PATH = FIGURES_DIR / "forecast_2026_full_year_plot.png"
MONTHLY_PROFILE_PLOT_PATH = FIGURES_DIR / "forecast_2026_monthly_profile_plot.png"
SEASONAL_PATTERN_PLOT_PATH = FIGURES_DIR / "forecast_2026_seasonal_pattern_plot.png"
COMPARISON_PLOT_PATH = FIGURES_DIR / "forecast_2026_table3_vs_table8_comparison_plot.png"


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_2026_base_calendar() -> pd.DataFrame:
    forecast_df = pd.DataFrame({"datetime": FORECAST_INDEX})
    forecast_df["hour"] = forecast_df["datetime"].dt.hour
    forecast_df["day_of_week"] = forecast_df["datetime"].dt.dayofweek
    forecast_df["month"] = forecast_df["datetime"].dt.month
    forecast_df["day"] = forecast_df["datetime"].dt.day
    forecast_df["day_of_year"] = forecast_df["datetime"].dt.dayofyear
    forecast_df["year"] = forecast_df["datetime"].dt.year
    forecast_df["year_month"] = forecast_df["datetime"].dt.to_period("M")
    forecast_df["date_only"] = forecast_df["datetime"].dt.normalize()
    forecast_df["weekend"] = (forecast_df["day_of_week"] >= 5).astype(int)
    forecast_df["night"] = forecast_df["hour"].between(0, 6).astype(int)
    return forecast_df


def add_holiday_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    holiday_index = pd.to_datetime(sorted(TURKEY_PUBLIC_HOLIDAYS))
    holiday_set = set(holiday_index)
    df["is_public_holiday"] = df["date_only"].isin(holiday_set).astype(int)
    holiday_window_set = holiday_set | {day - pd.Timedelta(days=1) for day in holiday_set} | {
        day + pd.Timedelta(days=1) for day in holiday_set
    }
    df["is_holiday_window"] = df["date_only"].isin(holiday_window_set).astype(int)
    prev_day_holiday = df["date_only"].map(lambda value: int((value - pd.Timedelta(days=1)) in holiday_set))
    next_day_holiday = df["date_only"].map(lambda value: int((value + pd.Timedelta(days=1)) in holiday_set))
    prev_day_weekend = df["day_of_week"].eq(0).astype(int)
    next_day_weekend = df["day_of_week"].eq(4).astype(int)
    df["is_bridge_day"] = (
        (df["is_public_holiday"] == 0)
        & (df["weekend"] == 0)
        & (
            ((prev_day_holiday == 1) & (next_day_weekend == 1))
            | ((prev_day_weekend == 1) & (next_day_holiday == 1))
            | ((prev_day_holiday == 1) & (next_day_holiday == 1))
        )
    ).astype(int)
    return df


def add_weather_proxy_features(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    weighted_df = pd.read_csv(WEIGHTED_HDD_CDD_PATH, parse_dates=["datetime"])
    source_2025 = weighted_df.loc[weighted_df["datetime"].dt.year == 2025, ["datetime", "weighted_HDD", "weighted_CDD"]].copy()
    source_2025["month"] = source_2025["datetime"].dt.month
    source_2025["day"] = source_2025["datetime"].dt.day
    source_2025["hour"] = source_2025["datetime"].dt.hour
    source_2025 = source_2025.drop(columns=["datetime"])

    df = forecast_df.merge(source_2025, on=["month", "day", "hour"], how="left", validate="many_to_one")
    note = (
        "2026 weighted_HDD and weighted_CDD were generated using a seasonal-naive weather proxy: "
        "for each 2026 month-day-hour, the population-weighted HDD/CDD values from the same month-day-hour in 2025 "
        "were copied forward. This preserves the validated population-weighted temperature construction while avoiding "
        "any unavailable true 2026 weather inputs."
    )
    return df, note


def add_macro_proxy_features(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    econ = load_economic_data().sort_values("year_month").reset_index(drop=True)
    econ_2025 = econ.loc[econ["year_month"].dt.year == 2025, ["year_month", "PMI", "IR"]].copy()
    if len(econ_2025) != 12:
        raise ValueError("2025 monthly economic data is incomplete; cannot build 2026 macro proxy assumptions.")

    assumed_2026 = pd.DataFrame({"year_month": pd.period_range("2026-01", "2026-12", freq="M")})
    assumed_2026["PMI"] = econ_2025["PMI"].to_numpy()
    assumed_2026["IR"] = econ_2025["IR"].to_numpy()

    combined = pd.concat(
        [
            econ.loc[econ["year_month"] == pd.Period("2025-12", freq="M"), ["year_month", "PMI", "IR"]],
            assumed_2026,
        ],
        ignore_index=True,
    ).sort_values("year_month")

    combined["PMI_prev_month"] = combined["PMI"].shift(1)
    combined["IR_prev_month"] = combined["IR"].shift(1)

    proxy_map = combined.loc[combined["year_month"].dt.year == 2026, ["year_month", "PMI_prev_month", "IR_prev_month"]]
    df = forecast_df.merge(proxy_map, on="year_month", how="left", validate="many_to_one")
    note = (
        "2026 PMI_prev_month and IR_prev_month were generated using a seasonal-naive monthly macro scenario. "
        "Assumed monthly 2026 PMI and IR levels were set equal to the corresponding observed 2025 monthly values. "
        "Release alignment was preserved: January 2026 uses December 2025 observed values, while February-December 2026 "
        "use the assumed previous-month 2026 values."
    )
    return df, note


def add_cyclic_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * (df["day_of_year"] - 1) / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * (df["day_of_year"] - 1) / 365.25)
    return df


def add_table8_historical_2026_features(forecast_df: pd.DataFrame, base_df_2022_2025: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = forecast_df.copy()
    historical_df = base_df_2022_2025.copy().sort_values("datetime").reset_index(drop=True)
    historical_df["day"] = historical_df["datetime"].dt.day
    historical_df["sqrt_consumption_seasonal_diff_24h"] = (
        historical_df["sqrt_consumption"] - historical_df["sqrt_consumption"].shift(24)
    )
    source_2025 = historical_df.loc[historical_df["year"] == 2025].copy().reset_index(drop=True)

    direct_lookup = (
        source_2025[["month", "day", "hour", "sqrt_consumption"]]
        .rename(columns={"sqrt_consumption": "prev_year_same_month_day_hour_sqrt"})
    )
    df = df.merge(direct_lookup, on=["month", "day", "hour"], how="left", validate="many_to_one")

    prev_hour_lookup = source_2025[["datetime", "sqrt_consumption"]].copy()
    prev_hour_lookup["mapped_time"] = prev_hour_lookup["datetime"] + pd.Timedelta(hours=1)
    prev_hour_lookup["month"] = prev_hour_lookup["mapped_time"].dt.month
    prev_hour_lookup["day"] = prev_hour_lookup["mapped_time"].dt.day
    prev_hour_lookup["hour"] = prev_hour_lookup["mapped_time"].dt.hour
    prev_hour_lookup = prev_hour_lookup[
        ["month", "day", "hour", "sqrt_consumption"]
    ].rename(columns={"sqrt_consumption": "prev_year_same_month_day_prev_hour_sqrt"})
    df = df.merge(prev_hour_lookup, on=["month", "day", "hour"], how="left", validate="many_to_one")

    seasonal_lookup = (
        source_2025[["month", "day", "hour", "sqrt_consumption_seasonal_diff_24h"]]
        .rename(columns={"sqrt_consumption_seasonal_diff_24h": "prev_year_same_month_day_seasonal_diff_24h_sqrt"})
    )
    df = df.merge(seasonal_lookup, on=["month", "day", "hour"], how="left", validate="many_to_one")

    hour_dow_lookup = (
        source_2025.groupby(["hour", "day_of_week"], as_index=False)["sqrt_consumption"]
        .mean()
        .rename(columns={"sqrt_consumption": "prev_year_mean_sqrt_by_hour_day_of_week"})
    )
    df = df.merge(hour_dow_lookup, on=["hour", "day_of_week"], how="left", validate="many_to_one")

    hour_month_lookup = (
        source_2025.groupby(["hour", "month"], as_index=False)["sqrt_consumption"]
        .mean()
        .rename(columns={"sqrt_consumption": "prev_year_mean_sqrt_by_hour_month"})
    )
    df = df.merge(hour_month_lookup, on=["hour", "month"], how="left", validate="many_to_one")

    note = (
        "All Table 8 historical analog/profile features for 2026 were generated strictly from 2025 source information only. "
        "Exact previous-year analogs use the same month-day-hour keys from 2025; the previous-hour analog maps 2025 timestamps "
        "forward by one hour; the seasonal-difference analog uses 2025 sqrt_consumption minus its own 24-hour lag; and both grouped "
        "profile features use 2025-only grouped means. No 2022-2025 pooled grouped aggregation was used, and no predicted 2026 "
        "consumption was fed back into feature construction."
    )
    return df, note


def build_table3_training_dataset(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from xgboost_population_weighted_pipeline import MODEL_CONFIGS

    config = next(item for item in MODEL_CONFIGS if item["model_key"] == "table3_sarima_guided_population_weighted")
    dataset, feature_columns = build_model_dataset(base_df, config)
    train_df = dataset.loc[dataset["year"].isin(TRAIN_YEARS_FINAL)].copy()
    return train_df, feature_columns


def build_table8_training_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    from xgboost_population_weighted_nonrecursive_historical_experiment import add_nonrecursive_historical_features

    augmented_df = add_nonrecursive_historical_features(base_df)
    selected_columns = ["datetime", "year", "consumption", "sqrt_consumption", *EXPERIMENTAL_FEATURES]
    dataset = augmented_df[selected_columns].copy()
    dataset = dataset.apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name != "datetime"
        else column
    )
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    train_df = dataset.loc[dataset["year"].isin(TRAIN_YEARS_FINAL)].copy()
    train_df = train_df.loc[train_df["year"] >= 2023].copy()
    return train_df


def fit_table3_model(train_df: pd.DataFrame, feature_columns: list[str]) -> XGBRegressor:
    model = build_xgb_model("table3_sarima_guided_population_weighted")
    model.fit(train_df[feature_columns], train_df["consumption"])
    return model


def fit_table8_model(train_df: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(**FINAL_MODEL_PARAMS)
    model.fit(train_df[EXPERIMENTAL_FEATURES], train_df["sqrt_consumption"])
    return model


def forecast_table3_2026_full_year_recursive(
    table3_model: XGBRegressor, forecast_df: pd.DataFrame, base_df: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    history_log = {
        pd.Timestamp(row.datetime): float(row.log_consumption)
        for row in base_df.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for row in forecast_df.itertuples(index=False):
        current_time = pd.Timestamp(row.datetime)
        lag_1_time = current_time - pd.Timedelta(hours=1)
        lag_25_time = current_time - pd.Timedelta(hours=25)

        lag_1_value = float(history_log[lag_1_time])
        lag_25_value = float(history_log[lag_25_time])
        seasonal_diff_value = lag_1_value - lag_25_value

        X = pd.DataFrame(
            [
                {
                    "weighted_HDD": float(row.weighted_HDD),
                    "weighted_CDD": float(row.weighted_CDD),
                    "night": int(row.night),
                    "weekend": int(row.weekend),
                    "log_consumption_lag_1h": lag_1_value,
                    "log_consumption_seasonal_diff_24h": seasonal_diff_value,
                }
            ]
        )

        predicted_consumption = float(table3_model.predict(X)[0])
        predicted_consumption = max(predicted_consumption, 1e-6)
        predicted_log = float(np.log(predicted_consumption))
        history_log[current_time] = predicted_log

        rows.append(
            {
                "datetime": current_time,
                "forecast_consumption": predicted_consumption,
                "weighted_HDD": float(row.weighted_HDD),
                "weighted_CDD": float(row.weighted_CDD),
                "log_consumption_lag_1h_used": lag_1_value,
                "log_consumption_seasonal_diff_24h_used": seasonal_diff_value,
            }
        )

    note = (
        "Table 3 2026 forecast was generated recursively because the validated Table 3 model is a short-term one-step-ahead design. "
        "For each 2026 hour, weighted_HDD, weighted_CDD, night, and weekend came from the 2026 exogenous proxy frame, while "
        "log_consumption_lag_1h and log_consumption_seasonal_diff_24h were updated recursively from the latest available history. "
        "Initialization used observed 2025 log consumption values; therefore the 2025->2026 boundary is handled by using "
        "2025-12-31 23:00 for lag_1h at 2026-01-01 00:00 and 2025-12-30 23:00 for the 25-hour seasonal-difference boundary. "
        "After the boundary, all unavailable 2026 lag values were replaced by previously predicted 2026 values."
    )
    return pd.DataFrame(rows), note


def build_table3_proxy_actual_2026(forecast_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    source_2025 = (
        base_df.loc[base_df["year"] == 2025, ["datetime", "consumption", "log_consumption"]]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    source_2025["month"] = source_2025["datetime"].dt.month
    source_2025["day"] = source_2025["datetime"].dt.day
    source_2025["hour"] = source_2025["datetime"].dt.hour

    lookup = source_2025[["month", "day", "hour", "consumption", "log_consumption"]]
    proxy_df = forecast_df[
        ["datetime", "year", "month", "day", "hour", "weighted_HDD", "weighted_CDD", "night", "weekend"]
    ].merge(lookup, on=["month", "day", "hour"], how="left", validate="many_to_one")

    missing_proxy = int(proxy_df["consumption"].isna().sum()) + int(proxy_df["log_consumption"].isna().sum())
    if missing_proxy:
        raise ValueError(f"Table 3 operational proxy actual construction failed with {missing_proxy} missing values.")
    return proxy_df


def build_table3_operational_training_history(
    base_df: pd.DataFrame, proxy_actual_2026: pd.DataFrame, month_start: pd.Timestamp
) -> pd.DataFrame:
    base_history = base_df[
        [
            "datetime",
            "year",
            "month",
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
            "consumption",
            "log_consumption",
        ]
    ].copy()
    operational_parts = [base_history]
    if month_start.month > 1:
        proxy_history = proxy_actual_2026.loc[proxy_actual_2026["datetime"] < month_start].copy()
        operational_parts.append(
            proxy_history[
                [
                    "datetime",
                    "year",
                    "month",
                    "weighted_HDD",
                    "weighted_CDD",
                    "night",
                    "weekend",
                    "consumption",
                    "log_consumption",
                ]
            ]
        )

    history_df = pd.concat(operational_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    history_df["log_consumption_lag_1h"] = history_df["log_consumption"].shift(1)
    history_df["log_consumption_lag_25h"] = history_df["log_consumption"].shift(25)
    history_df["log_consumption_seasonal_diff_24h"] = (
        history_df["log_consumption_lag_1h"] - history_df["log_consumption_lag_25h"]
    )
    return history_df


def forecast_table3_2026_month_ahead_rolling(
    forecast_df: pd.DataFrame, base_df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, str]:
    proxy_actual_2026 = build_table3_proxy_actual_2026(forecast_df, base_df)
    rows: list[dict[str, object]] = []
    month_summaries: list[str] = []

    for month in sorted(forecast_df["month"].unique()):
        month_start = pd.Timestamp(f"{FORECAST_YEAR}-{month:02d}-01 00:00:00")
        history_df = build_table3_operational_training_history(base_df, proxy_actual_2026, month_start)
        train_df = history_df.dropna(subset=["consumption", *feature_columns]).copy()
        month_model = fit_table3_model(train_df, feature_columns)

        history_log = {
            pd.Timestamp(row.datetime): float(row.log_consumption)
            for row in history_df.loc[history_df["datetime"] < month_start].itertuples(index=False)
        }
        month_frame = forecast_df.loc[forecast_df["month"] == month].copy().sort_values("datetime")

        for row in month_frame.itertuples(index=False):
            current_time = pd.Timestamp(row.datetime)
            lag_1_time = current_time - pd.Timedelta(hours=1)
            lag_25_time = current_time - pd.Timedelta(hours=25)

            lag_1_value = float(history_log[lag_1_time])
            lag_25_value = float(history_log[lag_25_time])
            seasonal_diff_value = lag_1_value - lag_25_value

            X = pd.DataFrame(
                [
                    {
                        "weighted_HDD": float(row.weighted_HDD),
                        "weighted_CDD": float(row.weighted_CDD),
                        "night": int(row.night),
                        "weekend": int(row.weekend),
                        "log_consumption_lag_1h": lag_1_value,
                        "log_consumption_seasonal_diff_24h": seasonal_diff_value,
                    }
                ]
            )

            predicted_consumption = float(month_model.predict(X)[0])
            predicted_consumption = max(predicted_consumption, 1e-6)
            predicted_log = float(np.log(predicted_consumption))
            history_log[current_time] = predicted_log

            rows.append(
                {
                    "datetime": current_time,
                    "forecast_consumption": predicted_consumption,
                    "forecast_month": int(month),
                    "weighted_HDD": float(row.weighted_HDD),
                    "weighted_CDD": float(row.weighted_CDD),
                    "log_consumption_lag_1h_used": lag_1_value,
                    "log_consumption_seasonal_diff_24h_used": seasonal_diff_value,
                    "initialization_source": "observed_2025_year_end" if month == 1 else "proxy_actualized_previous_2026_month",
                }
            )

        month_summaries.append(
            f"- {month_start.strftime('%Y-%m')}: refit history through {(month_start - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}; "
            f"training rows={len(train_df)}; monthly initialization={'observed 2025 year-end history' if month == 1 else 'proxy actualized prior 2026 months derived from 2025 analogs'}"
        )

    note = (
        "Table 3 2026 forecast was redesigned as a month-ahead rolling operational forecast. "
        "At the start of each 2026 month, the model is refit using all history available up to the previous month-end. "
        "Within the forecast month, recursive one-step-ahead updates are still used for log_consumption_lag_1h and "
        "log_consumption_seasonal_diff_24h. However, predicted values from month M are not propagated into month M+1 initialization. "
        "Because true 2026 actual consumption is unavailable in the current project snapshot, the operational-refresh assumption was "
        "implemented using a transparent proxy: before forecasting month M+1, the completed month M history is reinitialized from the "
        "same month-day-hour observed 2025 consumption path rather than from month M predictions. "
        "Thus the design preserves monthly operational reset logic while avoiding autonomous full-year recursive carry-forward.\n"
        + "\n".join(month_summaries)
    )
    return pd.DataFrame(rows), note


def forecast_table8_2026(table8_model: XGBRegressor, forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    X = forecast_df[EXPERIMENTAL_FEATURES].copy()
    raw_predictions = table8_model.predict(X)
    forecast_consumption = invert_predictions("square_to_consumption", raw_predictions)
    output_df = pd.DataFrame(
        {
            "datetime": forecast_df["datetime"],
            "prediction_sqrt_consumption": raw_predictions,
            "forecast_consumption": forecast_consumption,
        }
    )
    note = (
        "Table 8 2026 forecast was generated as a non-recursive direct-batch forecast. "
        "All 2026 feature values were constructed before prediction using only exogenous 2026 proxy inputs and 2025 historical source information. "
        "No predicted 2026 consumption was fed back into any feature."
    )
    return output_df, note


def validate_forecast_output(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    issues: list[str] = []
    if len(df) != 8760:
        issues.append(f"Expected 8760 rows, found {len(df)}.")
    if df["datetime"].min() != pd.Timestamp("2026-01-01 00:00:00"):
        issues.append(f"Unexpected min datetime: {df['datetime'].min()}")
    if df["datetime"].max() != pd.Timestamp("2026-12-31 23:00:00"):
        issues.append(f"Unexpected max datetime: {df['datetime'].max()}")
    if df["datetime"].duplicated().any():
        issues.append("Duplicate timestamps detected.")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    null_counts = df[required_columns].isna().sum()
    null_columns = null_counts[null_counts > 0]
    if not null_columns.empty:
        issues.append(f"Missing values found in required columns: {null_columns.to_dict()}")
    return issues


def plot_forecasts(table3_df: pd.DataFrame, table8_df: pd.DataFrame) -> None:
    plot_df = table3_df[["datetime", "forecast_consumption"]].rename(columns={"forecast_consumption": "table3"})
    plot_df = plot_df.merge(
        table8_df[["datetime", "forecast_consumption"]].rename(columns={"forecast_consumption": "table8"}),
        on="datetime",
        how="inner",
    )
    plot_df["date"] = plot_df["datetime"].dt.normalize()
    daily_df = plot_df.groupby("date", as_index=False)[["table3", "table8"]].mean()

    plt.figure(figsize=(16, 6))
    plt.plot(daily_df["date"], daily_df["table3"], label="Table 3 Rolling Month-Ahead", linewidth=1.1)
    plt.plot(daily_df["date"], daily_df["table8"], label="Table 8 Non-Recursive Historical", linewidth=1.1)
    plt.title("2026 Full-Year Forecast (Daily Average)")
    plt.xlabel("Date")
    plt.ylabel("Forecast Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FULL_YEAR_PLOT_PATH, dpi=160)
    plt.close()

    monthly_df = plot_df.groupby(plot_df["datetime"].dt.month)[["table3", "table8"]].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_df["datetime"], monthly_df["table3"], marker="o", label="Table 3 Rolling Month-Ahead")
    plt.plot(monthly_df["datetime"], monthly_df["table8"], marker="o", label="Table 8 Non-Recursive Historical")
    plt.title("2026 Monthly Mean Forecast Profile")
    plt.xlabel("Month")
    plt.ylabel("Mean Forecast Consumption")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(MONTHLY_PROFILE_PLOT_PATH, dpi=160)
    plt.close()

    seasonal_df = plot_df.copy()
    seasonal_df["hour"] = seasonal_df["datetime"].dt.hour
    seasonal_df["season"] = np.select(
        [
            seasonal_df["datetime"].dt.month.isin([12, 1, 2]),
            seasonal_df["datetime"].dt.month.isin([3, 4, 5]),
            seasonal_df["datetime"].dt.month.isin([6, 7, 8]),
            seasonal_df["datetime"].dt.month.isin([9, 10, 11]),
        ],
        ["Winter", "Spring", "Summer", "Autumn"],
        default="Other",
    )
    season_hour = seasonal_df.groupby(["season", "hour"], as_index=False)[["table3", "table8"]].mean()
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, season in zip(axes, season_order):
        season_slice = season_hour.loc[season_hour["season"] == season]
        ax.plot(season_slice["hour"], season_slice["table3"], label="Table 3 Rolling", linewidth=1.1)
        ax.plot(season_slice["hour"], season_slice["table8"], label="Table 8", linewidth=1.1)
        ax.set_title(season)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean Forecast Consumption")
    axes[0].legend()
    fig.suptitle("2026 Seasonal Hourly Forecast Pattern")
    plt.tight_layout()
    plt.savefig(SEASONAL_PATTERN_PLOT_PATH, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df["table3"], plot_df["table8"], s=6, alpha=0.22)
    min_val = float(min(plot_df["table3"].min(), plot_df["table8"].min()))
    max_val = float(max(plot_df["table3"].max(), plot_df["table8"].max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Table 3 Forecast")
    plt.ylabel("Table 8 Forecast")
    plt.title("Table 3 vs Table 8 Forecast Comparison (2026 Hourly)")
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=160)
    plt.close()


def main() -> None:
    validate_dependencies_and_files()
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base_df, _ = build_population_weighted_base_dataset(TRAIN_YEARS_FINAL)
    forecast_calendar = build_2026_base_calendar()
    forecast_calendar = add_holiday_features(forecast_calendar)
    forecast_calendar, weather_note = add_weather_proxy_features(forecast_calendar)
    forecast_calendar, macro_note = add_macro_proxy_features(forecast_calendar)
    forecast_calendar = add_cyclic_features(forecast_calendar)
    forecast_calendar, table8_feature_audit_note = add_table8_historical_2026_features(forecast_calendar, base_df)

    table3_train_df, table3_feature_columns = build_table3_training_dataset(base_df)
    table8_train_df = build_table8_training_dataset(base_df)

    table8_model = fit_table8_model(table8_train_df)

    table3_full_recursive_model = fit_table3_model(table3_train_df, table3_feature_columns)
    table3_previous_recursive_df, previous_recursive_note = forecast_table3_2026_full_year_recursive(
        table3_full_recursive_model, forecast_calendar, base_df
    )
    table3_forecast_df, table3_recursive_note = forecast_table3_2026_month_ahead_rolling(
        forecast_calendar, base_df, table3_feature_columns
    )
    table8_forecast_df, table8_nonrecursive_note = forecast_table8_2026(table8_model, forecast_calendar)

    table3_issues = validate_forecast_output(table3_forecast_df, ["datetime", "forecast_consumption"])
    table3_previous_issues = validate_forecast_output(
        table3_previous_recursive_df, ["datetime", "forecast_consumption"]
    )
    table8_issues = validate_forecast_output(
        table8_forecast_df,
        ["datetime", "prediction_sqrt_consumption", "forecast_consumption"],
    )
    forecast_feature_issues = []
    required_forecast_features = [
        "weighted_HDD",
        "weighted_CDD",
        "PMI_prev_month",
        "IR_prev_month",
        *EXPERIMENTAL_FEATURES,
    ]
    missing_feature_columns = [column for column in required_forecast_features if column not in forecast_calendar.columns]
    if missing_feature_columns:
        forecast_feature_issues.append(f"Missing forecast feature columns: {missing_feature_columns}")
    null_feature_counts = forecast_calendar[required_forecast_features].isna().sum()
    null_feature_columns = null_feature_counts[null_feature_counts > 0]
    if not null_feature_columns.empty:
        forecast_feature_issues.append(f"Missing values in forecast features: {null_feature_columns.to_dict()}")
    if forecast_calendar["datetime"].duplicated().any():
        forecast_feature_issues.append("Duplicate timestamps found in 2026 feature frame.")
    if len(forecast_calendar) != 8760:
        forecast_feature_issues.append(f"Feature frame row count mismatch: {len(forecast_calendar)}")

    table3_forecast_df.to_csv(TABLE3_FORECAST_PATH, index=False)
    table3_previous_recursive_df.to_csv(TABLE3_PREVIOUS_FULL_RECURSIVE_PATH, index=False)
    table8_forecast_df.to_csv(TABLE8_FORECAST_PATH, index=False)

    table3_comparison_df = table3_forecast_df[
        ["datetime", "forecast_month", "forecast_consumption"]
    ].rename(columns={"forecast_consumption": "rolling_month_ahead_forecast"})
    table3_comparison_df = table3_comparison_df.merge(
        table3_previous_recursive_df[["datetime", "forecast_consumption"]].rename(
            columns={"forecast_consumption": "previous_full_year_recursive_forecast"}
        ),
        on="datetime",
        how="left",
        validate="one_to_one",
    )
    table3_comparison_df["difference_rolling_minus_previous_recursive"] = (
        table3_comparison_df["rolling_month_ahead_forecast"]
        - table3_comparison_df["previous_full_year_recursive_forecast"]
    )
    table3_comparison_df["abs_difference"] = (
        table3_comparison_df["difference_rolling_minus_previous_recursive"].abs()
    )
    table3_comparison_df.to_csv(TABLE3_COMPARISON_PATH, index=False)

    comparison_stats = table3_comparison_df["difference_rolling_minus_previous_recursive"].agg(
        ["mean", "min", "max"]
    )
    abs_comparison_stats = table3_comparison_df["abs_difference"].agg(["mean", "max"])
    monthly_comparison = (
        table3_comparison_df.groupby("forecast_month", as_index=False)[
            ["rolling_month_ahead_forecast", "previous_full_year_recursive_forecast", "difference_rolling_minus_previous_recursive"]
        ]
        .mean()
    )
    comparison_lines = [
        "Table 3 2026 Rolling Month-Ahead vs Previous Full-Year Recursive Comparison",
        "",
        "Old implementation:",
        previous_recursive_note,
        "",
        "New implementation:",
        table3_recursive_note,
        "",
        "Validation:",
        f"- New rolling forecast issues: {table3_issues if table3_issues else 'None'}",
        f"- Previous full-year recursive forecast issues: {table3_previous_issues if table3_previous_issues else 'None'}",
        "",
        "Difference summary (rolling minus previous recursive):",
        f"- Mean difference: {comparison_stats['mean']:.6f}",
        f"- Min difference: {comparison_stats['min']:.6f}",
        f"- Max difference: {comparison_stats['max']:.6f}",
        f"- Mean absolute difference: {abs_comparison_stats['mean']:.6f}",
        f"- Max absolute difference: {abs_comparison_stats['max']:.6f}",
        f"- Annual total difference: {table3_comparison_df['difference_rolling_minus_previous_recursive'].sum():.6f}",
        "",
        "Monthly mean differences:",
        monthly_comparison.to_string(index=False),
    ]
    save_text(TABLE3_COMPARISON_NOTE_PATH, "\n".join(comparison_lines))

    table3_note_lines = [
        "Table 3 Forecast 2026 Methodology Note",
        "",
        "Model: Table 3 SARIMA-Guided Population-Weighted",
        "Target: consumption",
        f"Training period used for final forecast refit: {TRAIN_YEARS_FINAL}",
        "Feature set: weighted_HDD, weighted_CDD, night, weekend, log_consumption_lag_1h, log_consumption_seasonal_diff_24h",
        "",
        "Rolling month-ahead forecast design:",
        table3_recursive_note,
        "",
        "Exogenous input handling:",
        weather_note,
        "",
        "Validation checks:",
        f"- Forecast row count: {len(table3_forecast_df)}",
        f"- Datetime min/max: {table3_forecast_df['datetime'].min()} / {table3_forecast_df['datetime'].max()}",
        f"- Duplicate timestamp count: {int(table3_forecast_df['datetime'].duplicated().sum())}",
        f"- Issues: {table3_issues if table3_issues else 'None'}",
        "",
        "Previous implementation retained for comparison:",
        f"- Saved full-year recursive output: {TABLE3_PREVIOUS_FULL_RECURSIVE_PATH}",
        f"- Comparison note: {TABLE3_COMPARISON_NOTE_PATH}",
    ]
    save_text(TABLE3_NOTE_PATH, "\n".join(table3_note_lines))

    table8_note_lines = [
        "Table 8 Forecast 2026 Methodology Note",
        "",
        "Model: Table 8 Non-Recursive Historical Population-Weighted Forecast",
        "Target: sqrt_consumption",
        f"Training period used for final forecast refit: {TRAIN_YEARS_FINAL} (effective historical-analog rows 2023-2025)",
        "Forecast mode: non-recursive direct-batch",
        "",
        "Exogenous input handling:",
        weather_note,
        macro_note,
        "",
        "Historical feature rule for 2026:",
        table8_feature_audit_note,
        "",
        "Nonrecursive note:",
        table8_nonrecursive_note,
        "",
        "Validation checks:",
        f"- Forecast row count: {len(table8_forecast_df)}",
        f"- Datetime min/max: {table8_forecast_df['datetime'].min()} / {table8_forecast_df['datetime'].max()}",
        f"- Duplicate timestamp count: {int(table8_forecast_df['datetime'].duplicated().sum())}",
        f"- Feature frame issues: {forecast_feature_issues if forecast_feature_issues else 'None'}",
        f"- Forecast output issues: {table8_issues if table8_issues else 'None'}",
    ]
    save_text(TABLE8_NOTE_PATH, "\n".join(table8_note_lines))

    table8_audit_lines = [
        "Table 8 Forecast 2026 Feature Generation Audit",
        "",
        "Historical analog/profile source year:",
        "- All 2026 historical analog/profile features use 2025 source information only.",
        "- No 2022-2025 grouped aggregation was used for grouped profile features.",
        "",
        "Timeline:",
        "- Exact analogs: 2026 month-day-hour keys map to 2025 month-day-hour sqrt_consumption values.",
        "- Previous-hour analogs: 2025 timestamps were shifted forward by one hour before month-day-hour mapping.",
        "- Seasonal-difference analogs: 2025 sqrt_consumption minus its own 24-hour lag.",
        "- Grouped profiles: 2025-only mean sqrt_consumption by (hour, day_of_week) and by (hour, month).",
        "",
        "Future information usage:",
        "- No future 2026 actual consumption was used.",
        "- No predicted 2026 consumption was used in any feature.",
        "- Forecast is non-recursive and direct-batch.",
        "",
        "Feature completeness / missing-value check:",
        f"- Issues: {forecast_feature_issues if forecast_feature_issues else 'None'}",
    ]
    save_text(TABLE8_AUDIT_PATH, "\n".join(table8_audit_lines))

    plot_forecasts(table3_forecast_df, table8_forecast_df)

    comparison_df = table3_forecast_df[["datetime", "forecast_consumption"]].rename(columns={"forecast_consumption": "table3_forecast"})
    comparison_df = comparison_df.merge(
        table8_forecast_df[["datetime", "forecast_consumption"]].rename(columns={"forecast_consumption": "table8_forecast"}),
        on="datetime",
        how="inner",
    )
    comparison_df["difference_table3_minus_table8"] = comparison_df["table3_forecast"] - comparison_df["table8_forecast"]

    summary_lines = [
        "Final 2026 Forecast Model Comparison",
        "",
        "Model 1: Table 3 SARIMA-Guided Population-Weighted",
        "- Short-term model extended to 2026 via month-ahead rolling operational forecasts.",
        "- Information set: 2026 exogenous weather proxy + recursive within-month lag updates + monthly operational reset.",
        "- Between forecast months, predicted values are not carried forward; monthly reinitialization uses externally refreshed history assumptions.",
        "",
        "Model 2: Table 8 Non-Recursive Historical Population-Weighted Forecast",
        "- Long-term direct-batch model with locked Model C hyperparameters.",
        "- Information set: 2026 exogenous proxies + 2025-only historical analog/profile features.",
        "- No recursive feedback and no 2026 actual consumption usage.",
        "",
        "Methodological comparison:",
        "- Table 3 is recursive only within each forecast month; it no longer runs as an autonomous full-year recursive simulation.",
        "- Table 8 is non-recursive and direct-batch; each 2026 hour is predicted from exogenous inputs and 2025 historical structure only.",
        "- Table 3 reflects one-step-ahead SARIMA-guided persistence logic.",
        "- Table 8 reflects historical-analog long-term structure with population-weighted weather and locked Model C regularization.",
        "",
        "Exogenous assumptions:",
        f"- Weather: {weather_note}",
        f"- Macro: {macro_note}",
        "",
        "Validation summary:",
        f"- Table 3 output issues: {table3_issues if table3_issues else 'None'}",
        f"- Table 3 previous full-year recursive issues: {table3_previous_issues if table3_previous_issues else 'None'}",
        f"- Table 8 output issues: {table8_issues if table8_issues else 'None'}",
        f"- Table 8 feature frame issues: {forecast_feature_issues if forecast_feature_issues else 'None'}",
        f"- Final 2026 coverage: {comparison_df['datetime'].min()} to {comparison_df['datetime'].max()} ({len(comparison_df)} rows)",
        f"- Table 3 rolling vs previous recursive comparison file: {TABLE3_COMPARISON_PATH}",
        "",
        "Forecast interpretation notes:",
        "- Table 3 may still react strongly within a month because of recursive lag updates, but month-to-month forecast drift is no longer allowed to compound autonomously.",
        "- Table 8 may be smoother and more anchored to 2025 structural analogs and exogenous assumptions.",
        "- Differences between the two paths should be interpreted as information-set and forecast-logic differences, not only model-form differences.",
    ]
    save_text(SUMMARY_PATH, "\n".join(summary_lines))

    print("2026 forecast generation complete.")
    print(f"Table 3 rows: {len(table3_forecast_df)} | issues: {table3_issues if table3_issues else 'None'}")
    print(
        f"Table 3 previous full-year recursive rows: {len(table3_previous_recursive_df)} | issues: {table3_previous_issues if table3_previous_issues else 'None'}"
    )
    print(f"Table 8 rows: {len(table8_forecast_df)} | issues: {table8_issues if table8_issues else 'None'}")
    print(f"Feature frame issues: {forecast_feature_issues if forecast_feature_issues else 'None'}")
    print(
        f"Outputs: {TABLE3_FORECAST_PATH}, {TABLE3_COMPARISON_PATH}, {TABLE8_FORECAST_PATH}, {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
