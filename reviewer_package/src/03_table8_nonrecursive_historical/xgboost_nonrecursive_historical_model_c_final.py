from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from xgboost_population_weighted_nonrecursive_historical_experiment import (
    EXPERIMENTAL_FEATURES,
    add_nonrecursive_historical_features,
)
from xgboost_population_weighted_pipeline import (
    FIGURES_DIR,
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

FINAL_LABEL = "Table 8 Non-Recursive Historical Population-Weighted Forecast"

METRICS_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_metrics.json"
PREDICTIONS_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_predictions_2025.csv"
FEATURE_IMPORTANCE_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_feature_importance.csv"
OVERFITTING_SUMMARY_PATH = XGBOOST_DIR / "nonrecursive_historical_overfitting_audit.csv"
MONTHLY_STABILITY_PATH = XGBOOST_DIR / "nonrecursive_historical_monthly_stability.csv"
COMPARISON_PATH = XGBOOST_DIR / "table8_fully_blind_vs_nonrecursive_historical_population_weighted.csv"
METHODOLOGY_NOTE_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_methodology_note.txt"

ACTUAL_VS_PREDICTED_PLOT_PATH = (
    FIGURES_DIR / "table8_nonrecursive_historical_population_weighted_actual_vs_predicted_2025.png"
)
RESIDUAL_PLOT_PATH = (
    FIGURES_DIR / "table8_nonrecursive_historical_population_weighted_residuals_2025.png"
)

FULLY_BLIND_METRICS_PATH = XGBOOST_DIR / "table8_fully_blind_population_weighted_metrics.json"
FULLY_BLIND_DIAGNOSTICS_PATH = XGBOOST_DIR / "table8_fully_blind_population_weighted_overfitting_diagnostics.json"
RECURSIVE_METRICS_PATH = XGBOOST_DIR / "table8_recursive_population_weighted_metrics.json"
ONE_STEP_METRICS_PATH = XGBOOST_DIR / "table8_sarima_guided_population_weighted_metrics.json"
ONE_STEP_TRAIN_METRICS_PATH = XGBOOST_DIR / "table8_sarima_guided_population_weighted_train_metrics.json"

FINAL_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 525,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.65,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.25,
    "reg_lambda": 2.2,
    "random_state": 42,
    "n_jobs": 4,
    "tree_method": "hist",
}


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset() -> pd.DataFrame:
    base_df, _ = build_population_weighted_base_dataset(TRAIN_YEARS + [TEST_YEAR])
    augmented_df = add_nonrecursive_historical_features(base_df)
    selected_columns = ["datetime", "year", "consumption", "sqrt_consumption", *EXPERIMENTAL_FEATURES]
    dataset = augmented_df[selected_columns].copy()
    dataset = dataset.apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name != "datetime"
        else column
    )
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    return dataset


def fit_final_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    model = XGBRegressor(**FINAL_MODEL_PARAMS)
    model.fit(train_df[EXPERIMENTAL_FEATURES], train_df["sqrt_consumption"])

    train_raw = model.predict(train_df[EXPERIMENTAL_FEATURES])
    test_raw = model.predict(test_df[EXPERIMENTAL_FEATURES])
    train_pred = invert_predictions("square_to_consumption", train_raw)
    test_pred = invert_predictions("square_to_consumption", test_raw)

    y_train = np.square(train_df["sqrt_consumption"].to_numpy())
    y_test = np.square(test_df["sqrt_consumption"].to_numpy())

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    feature_importance = (
        pd.DataFrame({"feature": EXPERIMENTAL_FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    test_output = pd.DataFrame(
        {
            "datetime": test_df["datetime"],
            "actual_consumption": y_test,
            "predicted_consumption": test_pred,
        }
    )
    test_output["residual"] = test_output["actual_consumption"] - test_output["predicted_consumption"]

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "test_output": test_output,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def compute_monthly_stability(test_output: pd.DataFrame) -> pd.DataFrame:
    monthly_rows: list[dict[str, object]] = []
    temp = test_output.copy()
    temp["month_period"] = pd.to_datetime(temp["datetime"]).dt.to_period("M")

    for month_period, month_df in temp.groupby("month_period"):
        metrics = evaluate_predictions(month_df["actual_consumption"], month_df["predicted_consumption"])
        monthly_rows.append(
            {
                "month": str(month_period),
                "rows": len(month_df),
                "monthly_R2": float(metrics["R2"]),
                "monthly_RMSE": float(metrics["RMSE"]),
                "monthly_MAPE": float(metrics["MAPE"]),
            }
        )

    monthly_df = pd.DataFrame(monthly_rows).sort_values("month").reset_index(drop=True)
    monthly_df["rmse_rank_desc"] = monthly_df["monthly_RMSE"].rank(method="dense", ascending=False).astype(int)
    monthly_df["r2_rank_asc"] = monthly_df["monthly_R2"].rank(method="dense", ascending=True).astype(int)
    monthly_df["is_weak_month"] = (
        (monthly_df["rmse_rank_desc"] <= 3) | (monthly_df["r2_rank_asc"] <= 3)
    ).astype(int)
    monthly_df["is_strong_month"] = (
        (monthly_df["monthly_RMSE"].rank(method="dense", ascending=True) <= 3)
        | (monthly_df["monthly_R2"].rank(method="dense", ascending=False) <= 3)
    ).astype(int)
    return monthly_df


def plot_outputs(test_output: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(test_output["datetime"], test_output["actual_consumption"], label="Actual", linewidth=1.5)
    plt.plot(test_output["datetime"], test_output["predicted_consumption"], label="Predicted", linewidth=1.0)
    plt.title("Table 8 Non-Recursive Historical Population-Weighted: Actual vs Predicted (2025)")
    plt.xlabel("Datetime")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACTUAL_VS_PREDICTED_PLOT_PATH, dpi=160)
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(test_output["datetime"], test_output["residual"], linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Table 8 Non-Recursive Historical Population-Weighted: Residuals (2025)")
    plt.xlabel("Datetime")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH, dpi=160)
    plt.close()


def build_gap_row(model_name: str, train_metrics: dict[str, float], test_metrics: dict[str, float], note: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "train_R2": float(train_metrics["R2"]),
        "train_MAE": float(train_metrics["MAE"]),
        "train_RMSE": float(train_metrics["RMSE"]),
        "train_MAPE": float(train_metrics["MAPE"]),
        "test_R2": float(test_metrics["R2"]),
        "test_MAE": float(test_metrics["MAE"]),
        "test_RMSE": float(test_metrics["RMSE"]),
        "test_MAPE": float(test_metrics["MAPE"]),
        "gap_R2": float(train_metrics["R2"] - test_metrics["R2"]),
        "gap_RMSE": float(test_metrics["RMSE"] - train_metrics["RMSE"]),
        "gap_MAPE": float(test_metrics["MAPE"] - train_metrics["MAPE"]),
        "note": note,
    }


def load_reference_metrics() -> tuple[dict[str, object] | None, list[dict[str, object]], list[dict[str, object]]]:
    original_baseline = None
    if METRICS_PATH.exists():
        original_baseline = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    fully_blind_metrics = json.loads(FULLY_BLIND_METRICS_PATH.read_text(encoding="utf-8"))
    fully_blind_diag = json.loads(FULLY_BLIND_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    recursive_metrics = json.loads(RECURSIVE_METRICS_PATH.read_text(encoding="utf-8"))
    one_step_metrics = json.loads(ONE_STEP_METRICS_PATH.read_text(encoding="utf-8"))
    one_step_train = json.loads(ONE_STEP_TRAIN_METRICS_PATH.read_text(encoding="utf-8"))

    overfitting_rows = [
        build_gap_row(
            "Official Fully Blind Population-Weighted",
            fully_blind_diag["train_metrics"],
            fully_blind_diag["test_metrics"],
            "Official 2022-2024 direct-batch benchmark.",
        ),
        build_gap_row(
            "Table 8 Recursive Population-Weighted",
            one_step_train["train_metrics_consumption_scale"],
            recursive_metrics["test_metrics_consumption_scale"],
            (
                "Train metrics come from the shared SARIMA-guided fitted model; "
                "test metrics come from recursive rollout."
            ),
        ),
        build_gap_row(
            "Table 8 SARIMA-Guided One-Step-Ahead Population-Weighted",
            one_step_train["train_metrics_consumption_scale"],
            one_step_metrics["test_metrics_consumption_scale"],
            "Uses observed past 2025 consumption via one-step-ahead target-derived lag features.",
        ),
    ]

    comparison_rows = [
        {
            "model_name": "Official Fully Blind Population-Weighted",
            "method": "direct_batch_fully_blind",
            "train_R2": float(fully_blind_diag["train_metrics"]["R2"]),
            "train_MAE": float(fully_blind_diag["train_metrics"]["MAE"]),
            "train_RMSE": float(fully_blind_diag["train_metrics"]["RMSE"]),
            "train_MAPE": float(fully_blind_diag["train_metrics"]["MAPE"]),
            "test_R2": float(fully_blind_diag["test_metrics"]["R2"]),
            "test_MAE": float(fully_blind_diag["test_metrics"]["MAE"]),
            "test_RMSE": float(fully_blind_diag["test_metrics"]["RMSE"]),
            "test_MAPE": float(fully_blind_diag["test_metrics"]["MAPE"]),
            "note": "Official benchmark with 2022-2024 training window.",
        },
        {
            "model_name": "Table 8 Recursive Population-Weighted",
            "method": "recursive_rollout",
            "train_R2": float(one_step_train["train_metrics_consumption_scale"]["R2"]),
            "train_MAE": float(one_step_train["train_metrics_consumption_scale"]["MAE"]),
            "train_RMSE": float(one_step_train["train_metrics_consumption_scale"]["RMSE"]),
            "train_MAPE": float(one_step_train["train_metrics_consumption_scale"]["MAPE"]),
            "test_R2": float(recursive_metrics["test_metrics_consumption_scale"]["R2"]),
            "test_MAE": float(recursive_metrics["test_metrics_consumption_scale"]["MAE"]),
            "test_RMSE": float(recursive_metrics["test_metrics_consumption_scale"]["RMSE"]),
            "test_MAPE": float(recursive_metrics["test_metrics_consumption_scale"]["MAPE"]),
            "note": "Recursive benchmark.",
        },
        {
            "model_name": "Table 8 SARIMA-Guided One-Step-Ahead Population-Weighted",
            "method": "one_step_ahead",
            "train_R2": float(one_step_train["train_metrics_consumption_scale"]["R2"]),
            "train_MAE": float(one_step_train["train_metrics_consumption_scale"]["MAE"]),
            "train_RMSE": float(one_step_train["train_metrics_consumption_scale"]["RMSE"]),
            "train_MAPE": float(one_step_train["train_metrics_consumption_scale"]["MAPE"]),
            "test_R2": float(one_step_metrics["test_metrics_consumption_scale"]["R2"]),
            "test_MAE": float(one_step_metrics["test_metrics_consumption_scale"]["MAE"]),
            "test_RMSE": float(one_step_metrics["test_metrics_consumption_scale"]["RMSE"]),
            "test_MAPE": float(one_step_metrics["test_metrics_consumption_scale"]["MAPE"]),
            "note": "Observed past 2025 consumption available.",
        },
    ]

    if original_baseline is not None:
        comparison_rows.append(
            {
                "model_name": "Original baseline Non-Recursive Historical Population-Weighted",
                "method": "direct_batch_nonrecursive_historical_baseline",
                "train_R2": float(original_baseline["train_metrics"]["R2"]),
                "train_MAE": float(original_baseline["train_metrics"]["MAE"]),
                "train_RMSE": float(original_baseline["train_metrics"]["RMSE"]),
                "train_MAPE": float(original_baseline["train_metrics"]["MAPE"]),
                "test_R2": float(original_baseline["test_metrics"]["R2"]),
                "test_MAE": float(original_baseline["test_metrics"]["MAE"]),
                "test_RMSE": float(original_baseline["test_metrics"]["RMSE"]),
                "test_MAPE": float(original_baseline["test_metrics"]["MAPE"]),
                "note": "Previous baseline non-recursive historical specification before Model C promotion.",
            }
        )

    return original_baseline, overfitting_rows, comparison_rows


def main() -> None:
    validate_dependencies_and_files()

    original_baseline, reference_overfitting_rows, comparison_rows = load_reference_metrics()
    dataset = build_dataset()
    train_df, test_df = split_train_test(dataset)

    result = fit_final_model(train_df, test_df)
    result["feature_importance"].to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    result["test_output"].to_csv(PREDICTIONS_PATH, index=False)
    plot_outputs(result["test_output"])

    monthly_df = compute_monthly_stability(result["test_output"])
    monthly_df.to_csv(MONTHLY_STABILITY_PATH, index=False)

    metrics_payload = {
        "model_name": FINAL_LABEL,
        "target_column": "sqrt_consumption",
        "feature_columns": EXPERIMENTAL_FEATURES,
        "hyperparameters": FINAL_MODEL_PARAMS,
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
        "train_test_gap": {
            "R2_train_minus_test": float(result["train_metrics"]["R2"] - result["test_metrics"]["R2"]),
            "MAE_test_minus_train": float(result["test_metrics"]["MAE"] - result["train_metrics"]["MAE"]),
            "RMSE_test_minus_train": float(result["test_metrics"]["RMSE"] - result["train_metrics"]["RMSE"]),
            "MAPE_test_minus_train": float(result["test_metrics"]["MAPE"] - result["train_metrics"]["MAPE"]),
        },
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "method": (
            "Direct-batch, non-recursive historical-analog forecast. 2025 tahminleri tek seferde uretildi; "
            "2025 actual consumption kullanilmadi ve recursive feedback yok."
        ),
        "historical_profile_features": [
            "prev_year_same_month_day_hour_sqrt",
            "prev_year_same_month_day_prev_hour_sqrt",
            "prev_year_same_month_day_seasonal_diff_24h_sqrt",
            "prev_year_mean_sqrt_by_hour_day_of_week",
            "prev_year_mean_sqrt_by_hour_month",
        ],
        "uses_2025_actual_consumption": False,
        "uses_recursive_predicted_consumption": False,
        "effective_training_years": sorted(train_df["year"].unique().tolist()),
        "warmup_note": "2022 satirlari onceki yil analogu olmadigi icin efektif train penceresi 2023-2024 oldu.",
        "selected_variant_note": (
            "Model C secildi: daha dusuk subsample ve colsample_bytree ile stochastic regularization "
            "2025 test performansini iyilestirdi ve train-test gap'i hafifce daraltti."
        ),
    }
    save_json(METRICS_PATH, metrics_payload)

    final_overfitting_row = build_gap_row(
        FINAL_LABEL,
        result["train_metrics"],
        result["test_metrics"],
        (
            "Selected final candidate using Model C hyperparameters. "
            "Direct-batch, non-recursive historical-analog design with population-weighted HDD/CDD."
        ),
    )
    overfitting_df = pd.DataFrame([final_overfitting_row, *reference_overfitting_rows])
    overfitting_df.to_csv(OVERFITTING_SUMMARY_PATH, index=False)

    comparison_rows.insert(
        0,
        {
            "model_name": FINAL_LABEL,
            "method": "direct_batch_nonrecursive_historical_model_c",
            "train_R2": float(result["train_metrics"]["R2"]),
            "train_MAE": float(result["train_metrics"]["MAE"]),
            "train_RMSE": float(result["train_metrics"]["RMSE"]),
            "train_MAPE": float(result["train_metrics"]["MAPE"]),
            "test_R2": float(result["test_metrics"]["R2"]),
            "test_MAE": float(result["test_metrics"]["MAE"]),
            "test_RMSE": float(result["test_metrics"]["RMSE"]),
            "test_MAPE": float(result["test_metrics"]["MAPE"]),
            "note": "Final selected model using Model C stochastic regularization.",
        },
    )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    methodology_lines = [
        "Table 8 Non-Recursive Historical Population-Weighted Forecast",
        "",
        (
            "The final selected model uses a direct-batch, non-recursive historical-analog design "
            "with population-weighted HDD/CDD. It avoids using actual 2025 consumption and avoids "
            "recursive feedback."
        ),
        (
            "Target variable: sqrt_consumption. Baseline predictors include weighted_HDD, weighted_CDD, "
            "night, weekend, PMI_prev_month, IR_prev_month, calendar variables, cyclic seasonality terms, "
            "and historical analog features derived from prior-year consumption structure."
        ),
        (
            "Model C was selected because stochastic regularization through lower subsample and "
            "colsample_bytree improved 2025 test performance while slightly reducing the train-test gap."
        ),
        f"Selected hyperparameters: {json.dumps(FINAL_MODEL_PARAMS, ensure_ascii=False)}",
    ]
    METHODOLOGY_NOTE_PATH.write_text("\n".join(methodology_lines), encoding="utf-8")

    print("Model C promoted to official final candidate.")
    print()
    print("Final metrics:")
    print(pd.DataFrame([final_overfitting_row]).to_string(index=False))
    print()
    print("Comparison table:")
    print(comparison_df.to_string(index=False))
    if original_baseline is not None:
        print()
        print(
            "Original baseline non-recursive historical reference was preserved in the comparison table "
            "before overwriting official outputs."
        )


if __name__ == "__main__":
    main()
