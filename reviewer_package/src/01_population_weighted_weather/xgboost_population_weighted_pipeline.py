from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - environment guard
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

from talep_tahmin_tek_dosya import (
    CONSUMPTION_FILES,
    PROJECT_ROOT,
    TEMPERATURE_FILES,
    build_analysis_frame,
)
from xgboost_train import TURKEY_PUBLIC_HOLIDAYS


TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
ALL_YEARS = TRAIN_YEARS + [TEST_YEAR]
RANDOM_STATE = 42

WEIGHTED_HDD_CDD_PATH = PROJECT_ROOT / "outputs" / "reports" / "population_weighted_hourly_hdd_cdd.csv"
XGBOOST_DIR = PROJECT_ROOT / "outputs" / "xgboost"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

VALIDATION_TXT_PATH = XGBOOST_DIR / "population_weighted_hdd_cdd_merge_validation.txt"
VALIDATION_CSV_PATH = XGBOOST_DIR / "population_weighted_hdd_cdd_validation_summary.csv"
COMPARISON_CSV_PATH = XGBOOST_DIR / "table8_long_term_comparison_population_weighted.csv"
COMPARISON_TXT_PATH = XGBOOST_DIR / "table8_long_term_comparison_population_weighted.txt"
TRAIN_SUMMARY_PATH = XGBOOST_DIR / "xgboost_population_weighted_training_summary.csv"
TEST_SUMMARY_PATH = XGBOOST_DIR / "xgboost_population_weighted_test_summary.csv"
RECURSIVE_SUMMARY_PATH = XGBOOST_DIR / "xgboost_population_weighted_recursive_summary.csv"

MODEL_CONFIGS = [
    {
        "model_key": "table3_intramonth_population_weighted",
        "label": "Table 3 Intramonth Population-Weighted",
        "description": "Table 3 kisa donem modeli; HDD/CDD yerine population-weighted HDD/CDD kullanir.",
        "target_column": "consumption",
        "feature_columns": ["weighted_HDD", "weighted_CDD", "night", "weekend"],
        "prediction_transform": "identity",
        "evaluation_mode": "direct_batch",
    },
    {
        "model_key": "table3_sarima_guided_population_weighted",
        "label": "Table 3 SARIMA-Guided Population-Weighted",
        "description": "Table 3 + SARIMA feature tasarimi; weighted_HDD/weighted_CDD kullanir.",
        "target_column": "consumption",
        "feature_columns": [
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
            "log_consumption_lag_1h",
            "log_consumption_seasonal_diff_24h",
        ],
        "prediction_transform": "identity",
        "evaluation_mode": "one_step_ahead",
    },
    {
        "model_key": "table8_sarima_guided_population_weighted",
        "label": "Table 8 SARIMA-Guided One-Step-Ahead Population-Weighted",
        "description": "Table 8 uzun donem benchmark modeli; weighted_HDD/weighted_CDD ve SARIMA-guided lag feature'lari kullanir.",
        "target_column": "sqrt_consumption",
        "feature_columns": [
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
            "PMI_prev_month",
            "IR_prev_month",
            "log_consumption_lag_1h",
            "log_consumption_seasonal_diff_24h",
        ],
        "prediction_transform": "square_to_consumption",
        "evaluation_mode": "one_step_ahead",
        "is_table8_comparison": True,
    },
    {
        "model_key": "table8_fully_blind_population_weighted",
        "label": "Table 8 Fully Blind Broad Future Forecast Population-Weighted",
        "description": "Table 8 fully blind uzun donem modeli; yalnizca exogenous ve calendar feature'lari kullanir.",
        "target_column": "sqrt_consumption",
        "feature_columns": [
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
        "prediction_transform": "square_to_consumption",
        "evaluation_mode": "fully_blind_direct_batch",
        "fully_blind": True,
        "is_table8_comparison": True,
    },
]


def validate_dependencies_and_files() -> None:
    if XGBRegressor is None:
        raise ImportError(
            "XGBoost population-weighted pipeline icin `xgboost` paketi gerekli."
        ) from XGBOOST_IMPORT_ERROR

    required_paths = [WEIGHTED_HDD_CDD_PATH]
    for year in ALL_YEARS:
        required_paths.extend([CONSUMPTION_FILES[year], TEMPERATURE_FILES[year]])

    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Population-weighted XGBoost pipeline icin gereken dosyalar eksik:\n"
            + "\n".join(missing_paths)
        )


def invert_predictions(prediction_transform: str, raw_predictions: np.ndarray) -> np.ndarray:
    if prediction_transform == "square_to_consumption":
        return np.square(np.clip(raw_predictions, a_min=0.0, a_max=None))
    return raw_predictions


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_series = pd.Series(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    nonzero_mask = y_true_series != 0
    mape = float(
        np.mean(
            np.abs(
                (y_true_series.loc[nonzero_mask] - y_pred_array[nonzero_mask])
                / y_true_series.loc[nonzero_mask]
            )
        )
        * 100
    )
    return {
        "R2": float(r2_score(y_true_series, y_pred_array)),
        "MAE": float(mean_absolute_error(y_true_series, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_series, y_pred_array))),
        "MAPE": mape,
    }


def build_population_weighted_base_dataset(years: list[int]) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = build_analysis_frame(years=years).sort_values("datetime").reset_index(drop=True)
    weighted_df = pd.read_csv(WEIGHTED_HDD_CDD_PATH, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)

    merged = raw_df.merge(weighted_df, on="datetime", how="left", validate="one_to_one")

    for column in ["weighted_HDD", "weighted_CDD"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    missing_weighted_hdd = int(merged["weighted_HDD"].isna().sum())
    missing_weighted_cdd = int(merged["weighted_CDD"].isna().sum())
    if missing_weighted_hdd or missing_weighted_cdd:
        raise ValueError(
            f"Merge sonrasi weighted HDD/CDD null degerleri var: HDD={missing_weighted_hdd}, CDD={missing_weighted_cdd}"
        )

    merged["year_month"] = merged["datetime"].dt.to_period("M")
    merged["date_only"] = merged["datetime"].dt.normalize()
    merged["day_of_year"] = merged["datetime"].dt.dayofyear
    merged["log_consumption_lag_1h"] = merged["log_consumption"].shift(1)
    merged["log_consumption_lag_25h"] = merged["log_consumption"].shift(25)
    merged["log_consumption_seasonal_diff_24h"] = (
        merged["log_consumption_lag_1h"] - merged["log_consumption_lag_25h"]
    )

    holiday_index = pd.to_datetime(sorted(TURKEY_PUBLIC_HOLIDAYS))
    holiday_set = set(holiday_index)
    merged["is_public_holiday"] = merged["date_only"].isin(holiday_set).astype(int)
    holiday_window_set = holiday_set | {day - pd.Timedelta(days=1) for day in holiday_set} | {
        day + pd.Timedelta(days=1) for day in holiday_set
    }
    merged["is_holiday_window"] = merged["date_only"].isin(holiday_window_set).astype(int)
    prev_day_holiday = merged["date_only"].map(
        lambda value: int((value - pd.Timedelta(days=1)) in holiday_set)
    )
    next_day_holiday = merged["date_only"].map(
        lambda value: int((value + pd.Timedelta(days=1)) in holiday_set)
    )
    prev_day_weekend = merged["day_of_week"].eq(0).astype(int)
    next_day_weekend = merged["day_of_week"].eq(4).astype(int)
    merged["is_bridge_day"] = (
        (merged["is_public_holiday"] == 0)
        & (merged["weekend"] == 0)
        & (
            ((prev_day_holiday == 1) & (next_day_weekend == 1))
            | ((prev_day_weekend == 1) & (next_day_holiday == 1))
            | ((prev_day_holiday == 1) & (next_day_holiday == 1))
        )
    ).astype(int)

    for column in ("PMI", "IR", "CUR"):
        monthly_series = merged.groupby("year_month")[column].first().sort_index()
        previous_month_series = monthly_series.shift(1)
        merged[f"{column}_prev_month"] = merged["year_month"].map(previous_month_series)

    comparison_rows = []
    if {"HDD", "CDD", "weighted_HDD", "weighted_CDD"}.issubset(merged.columns):
        comparison_rows = [
            {
                "comparison": "HDD_vs_weighted_HDD",
                "correlation": float(merged["HDD"].corr(merged["weighted_HDD"])),
                "mean_original": float(merged["HDD"].mean()),
                "mean_weighted": float(merged["weighted_HDD"].mean()),
                "mean_abs_diff": float(np.abs(merged["HDD"] - merged["weighted_HDD"]).mean()),
            },
            {
                "comparison": "CDD_vs_weighted_CDD",
                "correlation": float(merged["CDD"].corr(merged["weighted_CDD"])),
                "mean_original": float(merged["CDD"].mean()),
                "mean_weighted": float(merged["weighted_CDD"].mean()),
                "mean_abs_diff": float(np.abs(merged["CDD"] - merged["weighted_CDD"]).mean()),
            },
        ]
        pd.DataFrame(comparison_rows).to_csv(VALIDATION_CSV_PATH, index=False)

    validation_lines = [
        "Population-Weighted HDD/CDD Merge Validation",
        "",
        f"Datetime coverage min: {merged['datetime'].min()}",
        f"Datetime coverage max: {merged['datetime'].max()}",
        f"Expected hours from weighted file: {len(weighted_df)}",
        f"Merged row count: {len(merged)}",
        f"Missing weighted_HDD after merge: {missing_weighted_hdd}",
        f"Missing weighted_CDD after merge: {missing_weighted_cdd}",
        "",
        "weighted_HDD summary statistics:",
        merged["weighted_HDD"].describe().to_string(),
        "",
        "weighted_CDD summary statistics:",
        merged["weighted_CDD"].describe().to_string(),
    ]
    if comparison_rows:
        validation_lines.extend(
            [
                "",
                "Old HDD/CDD vs weighted HDD/CDD comparison:",
                pd.DataFrame(comparison_rows).to_string(index=False),
            ]
        )
    VALIDATION_TXT_PATH.write_text("\n".join(validation_lines), encoding="utf-8")

    return merged, {
        "validation_txt_path": str(VALIDATION_TXT_PATH),
        "validation_csv_path": str(VALIDATION_CSV_PATH),
    }


def build_model_dataset(base_df: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = list(config["feature_columns"])
    target_column = str(config["target_column"])
    selected_columns = list(dict.fromkeys(["datetime", "year", "consumption", target_column, *feature_columns]))
    dataset = base_df[selected_columns].copy()
    numeric_columns = list(dict.fromkeys(["consumption", target_column, *feature_columns]))
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    return dataset, feature_columns


def split_train_test(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = dataset.loc[dataset["year"].isin(TRAIN_YEARS)].copy()
    test_df = dataset.loc[dataset["year"] == TEST_YEAR].copy()
    return train_df, test_df


def build_xgb_model(model_key: str) -> XGBRegressor:
    if model_key == "table3_intramonth_population_weighted":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    if model_key == "table3_sarima_guided_population_weighted":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=425,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.2,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    if model_key == "table8_sarima_guided_population_weighted":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=450,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.2,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    if model_key == "table8_fully_blind_population_weighted":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=525,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.25,
            reg_lambda=2.2,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
    )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_models(base_df: pd.DataFrame) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []

    for config in MODEL_CONFIGS:
        model_key = str(config["model_key"])
        target_column = str(config["target_column"])
        prediction_transform = str(config["prediction_transform"])

        dataset, feature_columns = build_model_dataset(base_df, config)
        train_df, _ = split_train_test(dataset)
        model = build_xgb_model(model_key)
        model.fit(train_df[feature_columns], train_df[target_column])

        raw_train_predictions = model.predict(train_df[feature_columns])
        train_predictions_consumption = invert_predictions(prediction_transform, raw_train_predictions)
        y_train_consumption = (
            np.square(train_df[target_column]) if prediction_transform == "square_to_consumption" else train_df[target_column].to_numpy()
        )
        train_metrics = evaluate_predictions(y_train_consumption, train_predictions_consumption)

        feature_importance = (
            pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        train_predictions_df = train_df.copy()
        train_predictions_df["prediction_target_scale"] = raw_train_predictions
        train_predictions_df["prediction_consumption"] = train_predictions_consumption

        model_path = MODELS_DIR / f"{model_key}_xgboost_model.json"
        metadata_path = MODELS_DIR / f"{model_key}_xgboost_metadata.json"
        feature_importance_path = XGBOOST_DIR / f"{model_key}_feature_importance.csv"
        train_prediction_path = XGBOOST_DIR / f"{model_key}_train_predictions.csv"
        train_metrics_path = XGBOOST_DIR / f"{model_key}_train_metrics.json"

        model.save_model(model_path)
        feature_importance.to_csv(feature_importance_path, index=False)
        train_predictions_df.to_csv(train_prediction_path, index=False)
        save_json(
            train_metrics_path,
            {
                "model_key": model_key,
                "label": config["label"],
                "target_column": target_column,
                "feature_columns": feature_columns,
                "train_metrics_consumption_scale": train_metrics,
                "row_count": len(train_df),
                "first_timestamp": str(train_df["datetime"].min()),
                "last_timestamp": str(train_df["datetime"].max()),
                "population_weighted_temperature_features": True,
            },
        )
        save_json(
            metadata_path,
            {
                "model_key": model_key,
                "label": config["label"],
                "description": config["description"],
                "target_column": target_column,
                "feature_columns": feature_columns,
                "prediction_transform": prediction_transform,
                "train_years": TRAIN_YEARS,
                "test_year": TEST_YEAR,
                "train_prediction_path": str(train_prediction_path),
                "feature_importance_path": str(feature_importance_path),
                "train_metrics_path": str(train_metrics_path),
                "model_path": str(model_path),
                "temperature_feature_variant": "population_weighted_hdd_cdd",
            },
        )

        summary_rows.append(
            {
                "model_key": model_key,
                "label": config["label"],
                "row_count": len(train_df),
                "feature_count": len(feature_columns),
                "train_R2": train_metrics["R2"],
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_MAPE": train_metrics["MAPE"],
                "train_metrics_path": str(train_metrics_path),
                "feature_importance_path": str(feature_importance_path),
                "model_path": str(model_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(TRAIN_SUMMARY_PATH, index=False)
    return summary_rows


def test_models(base_df: pd.DataFrame) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []

    for config in MODEL_CONFIGS:
        model_key = str(config["model_key"])
        target_column = str(config["target_column"])
        prediction_transform = str(config["prediction_transform"])
        feature_columns = list(config["feature_columns"])

        dataset, _ = build_model_dataset(base_df, config)
        _, test_df = split_train_test(dataset)

        model = XGBRegressor()
        model.load_model(MODELS_DIR / f"{model_key}_xgboost_model.json")

        raw_test_predictions = model.predict(test_df[feature_columns])
        test_predictions_consumption = invert_predictions(prediction_transform, raw_test_predictions)
        y_test_consumption = (
            np.square(test_df[target_column]) if prediction_transform == "square_to_consumption" else test_df[target_column].to_numpy()
        )
        test_metrics = evaluate_predictions(y_test_consumption, test_predictions_consumption)

        output_df = test_df.copy()
        output_df["prediction_target_scale"] = raw_test_predictions
        output_df["prediction_consumption"] = test_predictions_consumption

        prediction_path = XGBOOST_DIR / f"{model_key}_predictions_2025.csv"
        metrics_path = XGBOOST_DIR / f"{model_key}_metrics.json"
        output_df.to_csv(prediction_path, index=False)
        save_json(
            metrics_path,
            {
                "model_key": model_key,
                "label": config["label"],
                "target_column": target_column,
                "feature_columns": feature_columns,
                "test_metrics_consumption_scale": test_metrics,
                "row_count": len(test_df),
                "first_timestamp": str(test_df["datetime"].min()),
                "last_timestamp": str(test_df["datetime"].max()),
                "prediction_path": str(prediction_path),
                "method": config["evaluation_mode"],
                "population_weighted_temperature_features": True,
            },
        )

        summary_rows.append(
            {
                "model_key": model_key,
                "label": config["label"],
                "test_R2": test_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAPE": test_metrics["MAPE"],
                "metrics_path": str(metrics_path),
                "prediction_path": str(prediction_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(TEST_SUMMARY_PATH, index=False)
    return summary_rows


def recursive_forecast_table8(base_df: pd.DataFrame) -> dict[str, object]:
    config = next(item for item in MODEL_CONFIGS if item["model_key"] == "table8_sarima_guided_population_weighted")
    model_key = str(config["model_key"])
    target_column = str(config["target_column"])
    feature_columns = list(config["feature_columns"])

    dataset, _ = build_model_dataset(base_df, config)
    train_df, test_df = split_train_test(dataset)

    history_log_consumption = {
        pd.Timestamp(row.datetime): float(row.log_consumption)
        for row in base_df.loc[base_df["year"].isin(TRAIN_YEARS)].itertuples(index=False)
    }

    model = XGBRegressor()
    model.load_model(MODELS_DIR / f"{model_key}_xgboost_model.json")

    recursive_rows: list[dict[str, object]] = []
    for row in test_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        current_time = pd.Timestamp(row_series["datetime"])
        feature_row: dict[str, float] = {}

        for feature in feature_columns:
            if feature == "log_consumption_lag_1h":
                lag_1_time = current_time - pd.Timedelta(hours=1)
                feature_row[feature] = float(history_log_consumption[lag_1_time])
            elif feature == "log_consumption_seasonal_diff_24h":
                lag_1_time = current_time - pd.Timedelta(hours=1)
                lag_25_time = current_time - pd.Timedelta(hours=25)
                feature_row[feature] = float(history_log_consumption[lag_1_time] - history_log_consumption[lag_25_time])
            else:
                feature_row[feature] = float(row_series[feature])

        X = pd.DataFrame([feature_row], columns=feature_columns)
        predicted_target_scale = float(model.predict(X)[0])
        predicted_consumption = float(invert_predictions(config["prediction_transform"], np.array([predicted_target_scale]))[0])
        predicted_log_consumption = float(np.log(max(predicted_consumption, 1e-6)))
        history_log_consumption[current_time] = predicted_log_consumption

        recursive_rows.append(
            {
                "datetime": current_time,
                "consumption": float(row_series["consumption"]),
                target_column: float(row_series[target_column]),
                "prediction_target_scale": predicted_target_scale,
                "prediction_consumption": predicted_consumption,
            }
        )

    recursive_df = pd.DataFrame(recursive_rows)
    recursive_metrics = evaluate_predictions(recursive_df["consumption"], recursive_df["prediction_consumption"])
    prediction_path = XGBOOST_DIR / "table8_recursive_population_weighted_predictions_2025.csv"
    metrics_path = XGBOOST_DIR / "table8_recursive_population_weighted_metrics.json"
    recursive_df.to_csv(prediction_path, index=False)
    save_json(
        metrics_path,
        {
            "model_key": "table8_recursive_population_weighted",
            "label": "Table 8 Recursive Population-Weighted",
            "feature_columns": feature_columns,
            "test_metrics_consumption_scale": recursive_metrics,
            "row_count": len(recursive_df),
            "first_timestamp": str(recursive_df["datetime"].min()),
            "last_timestamp": str(recursive_df["datetime"].max()),
            "prediction_path": str(prediction_path),
            "method": (
                "Recursive forecast. 2025 boyunca log_consumption_lag_1h ve "
                "log_consumption_seasonal_diff_24h model tahminlerinden uretildi; "
                "weighted_HDD/weighted_CDD ve makro degiskenler exogenous olarak kullanildi."
            ),
            "population_weighted_temperature_features": True,
        },
    )
    summary = {
        "model_key": "table8_recursive_population_weighted",
        "label": "Table 8 Recursive Population-Weighted",
        "test_R2": recursive_metrics["R2"],
        "test_MAE": recursive_metrics["MAE"],
        "test_RMSE": recursive_metrics["RMSE"],
        "test_MAPE": recursive_metrics["MAPE"],
        "metrics_path": str(metrics_path),
        "prediction_path": str(prediction_path),
    }
    pd.DataFrame([summary]).to_csv(RECURSIVE_SUMMARY_PATH, index=False)
    return summary


def build_long_term_comparison() -> pd.DataFrame:
    metrics_specs = [
        (
            "Table 8 Fully Blind Broad Future Forecast",
            XGBOOST_DIR / "table8_fully_blind_population_weighted_metrics.json",
            "primary_long_term",
        ),
        (
            "Table 8 SARIMA-Guided One-Step-Ahead",
            XGBOOST_DIR / "table8_sarima_guided_population_weighted_metrics.json",
            "benchmark_one_step",
        ),
        (
            "Table 8 Recursive",
            XGBOOST_DIR / "table8_recursive_population_weighted_metrics.json",
            "benchmark_recursive",
        ),
    ]

    rows = []
    for display_name, path, role in metrics_specs:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        values = metrics["test_metrics_consumption_scale"]
        rows.append(
            {
                "model": display_name,
                "role": role,
                "R2": round(float(values["R2"]), 4),
                "MAE": round(float(values["MAE"]), 4),
                "RMSE": round(float(values["RMSE"]), 4),
                "MAPE": round(float(values["MAPE"]), 4),
                "method": metrics["method"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)

    fully_blind = comparison.loc[comparison["role"] == "primary_long_term"].iloc[0]
    one_step = comparison.loc[comparison["role"] == "benchmark_one_step"].iloc[0]
    recursive = comparison.loc[comparison["role"] == "benchmark_recursive"].iloc[0]

    lines = [
        "Table 8 Long-Term Comparison Using Population-Weighted HDD/CDD",
        "",
        comparison.to_string(index=False),
        "",
        "Interpretation:",
        (
            f"- Primary long-term model is the Table 8 Fully Blind Broad Future Forecast. "
            f"It achieves R2={fully_blind['R2']:.4f}, MAE={fully_blind['MAE']:.2f}, "
            f"RMSE={fully_blind['RMSE']:.2f}, MAPE={fully_blind['MAPE']:.2f}%."
        ),
        (
            f"- Table 8 SARIMA-guided one-step-ahead is numerically stronger "
            f"(R2={one_step['R2']:.4f}, RMSE={one_step['RMSE']:.2f}) because it can use "
            "observed past 2025 consumption through target-derived lag features."
        ),
        (
            f"- Table 8 recursive is weaker "
            f"(R2={recursive['R2']:.4f}, RMSE={recursive['RMSE']:.2f}) because recursive "
            "feedback accumulates forecast error over the horizon."
        ),
        (
            "- Final long-term interpretation should prioritize the fully blind model because it "
            "predicts the entire 2025 horizon in one batch and does not use any 2025 "
            "consumption-derived feature."
        ),
    ]
    COMPARISON_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return comparison


def plot_fully_blind_diagnostics(base_df: pd.DataFrame) -> dict[str, str]:
    config = next(item for item in MODEL_CONFIGS if item["model_key"] == "table8_fully_blind_population_weighted")
    model_key = str(config["model_key"])
    target_column = str(config["target_column"])
    feature_columns = list(config["feature_columns"])

    dataset, _ = build_model_dataset(base_df, config)
    train_df, test_df = split_train_test(dataset)

    model = XGBRegressor()
    model.load_model(MODELS_DIR / f"{model_key}_xgboost_model.json")

    train_raw = model.predict(train_df[feature_columns])
    test_raw = model.predict(test_df[feature_columns])
    train_pred = invert_predictions(config["prediction_transform"], train_raw)
    test_pred = invert_predictions(config["prediction_transform"], test_raw)

    y_train = np.square(train_df[target_column])
    y_test = np.square(test_df[target_column])

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)
    train_test_gap = {
        metric: float(train_metrics[metric] - test_metrics[metric])
        for metric in train_metrics
    }

    diagnostics_path = XGBOOST_DIR / "table8_fully_blind_population_weighted_overfitting_diagnostics.json"
    save_json(
        diagnostics_path,
        {
            "model_key": model_key,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_minus_test_gap": train_test_gap,
            "feature_columns": feature_columns,
            "temperature_feature_variant": "population_weighted_hdd_cdd",
        },
    )

    feature_importance = pd.read_csv(XGBOOST_DIR / f"{model_key}_feature_importance.csv")
    feature_importance_path = XGBOOST_DIR / "table8_fully_blind_population_weighted_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    test_output = pd.DataFrame(
        {
            "datetime": test_df["datetime"],
            "actual_consumption": y_test,
            "predicted_consumption": test_pred,
            "residual": y_test - test_pred,
        }
    )
    prediction_path = XGBOOST_DIR / "table8_fully_blind_population_weighted_predictions_2025.csv"
    test_output.to_csv(prediction_path, index=False)

    long_plot_path = FIGURES_DIR / "table8_fully_blind_population_weighted_actual_vs_predicted_2025.png"
    short_plot_path = FIGURES_DIR / "table8_fully_blind_population_weighted_actual_vs_predicted_2025_january.png"
    residual_plot_path = FIGURES_DIR / "table8_fully_blind_population_weighted_residuals_2025.png"
    importance_plot_path = FIGURES_DIR / "table8_fully_blind_population_weighted_feature_importance.png"

    plt.figure(figsize=(16, 6))
    plt.plot(test_output["datetime"], test_output["actual_consumption"], label="Actual", linewidth=1.5)
    plt.plot(test_output["datetime"], test_output["predicted_consumption"], label="Predicted", linewidth=1.0)
    plt.title("Table 8 Fully Blind Population-Weighted: Actual vs Predicted (2025)")
    plt.xlabel("Datetime")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(long_plot_path, dpi=160)
    plt.close()

    january_mask = test_output["datetime"].dt.to_period("M") == pd.Period("2025-01")
    january_df = test_output.loc[january_mask]
    plt.figure(figsize=(16, 6))
    plt.plot(january_df["datetime"], january_df["actual_consumption"], label="Actual", linewidth=1.5)
    plt.plot(january_df["datetime"], january_df["predicted_consumption"], label="Predicted", linewidth=1.0)
    plt.title("Table 8 Fully Blind Population-Weighted: Actual vs Predicted (2025-01)")
    plt.xlabel("Datetime")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(short_plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(test_output["datetime"], test_output["residual"], linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Table 8 Fully Blind Population-Weighted: Residuals Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(residual_plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    plot_importance = feature_importance.sort_values("importance", ascending=True)
    plt.barh(plot_importance["feature"], plot_importance["importance"])
    plt.title("Table 8 Fully Blind Population-Weighted: Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=160)
    plt.close()

    return {
        "diagnostics_path": str(diagnostics_path),
        "prediction_path": str(prediction_path),
        "feature_importance_path": str(feature_importance_path),
        "long_plot_path": str(long_plot_path),
        "short_plot_path": str(short_plot_path),
        "residual_plot_path": str(residual_plot_path),
        "importance_plot_path": str(importance_plot_path),
    }


def main() -> None:
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    validate_dependencies_and_files()
    base_df, validation_metadata = build_population_weighted_base_dataset(ALL_YEARS)

    train_summary = train_models(base_df)
    test_summary = test_models(base_df)
    recursive_summary = recursive_forecast_table8(base_df)
    comparison = build_long_term_comparison()
    diagnostics_outputs = plot_fully_blind_diagnostics(base_df)

    print("Population-weighted XGBoost pipeline tamamlandi.")
    print(f"Validation TXT: {validation_metadata['validation_txt_path']}")
    print(f"Validation CSV: {validation_metadata['validation_csv_path']}")
    print(f"Training summary: {TRAIN_SUMMARY_PATH}")
    print(f"Test summary: {TEST_SUMMARY_PATH}")
    print(f"Recursive summary: {RECURSIVE_SUMMARY_PATH}")
    print(f"Long-term comparison: {COMPARISON_CSV_PATH}")
    print(comparison.to_string(index=False))
    print()
    print("Final fully blind diagnostics:")
    for key, value in diagnostics_outputs.items():
        print(f"- {key}: {value}")
    print()
    print("Trained models:")
    for row in train_summary:
        print(f"- {row['label']} | train_R2={row['train_R2']:.4f}")
    print(f"- {recursive_summary['label']} | test_R2={recursive_summary['test_R2']:.4f}")
    for row in test_summary:
        print(f"- {row['label']} | test_R2={row['test_R2']:.4f} | test_RMSE={row['test_RMSE']:.2f}")


if __name__ == "__main__":
    main()
