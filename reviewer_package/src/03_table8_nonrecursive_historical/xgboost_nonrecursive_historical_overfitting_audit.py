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
    build_xgb_model,
    evaluate_predictions,
    invert_predictions,
    split_train_test,
    validate_dependencies_and_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent

OVERFITTING_AUDIT_PATH = XGBOOST_DIR / "nonrecursive_historical_overfitting_audit.csv"
REGULARIZATION_STUDY_PATH = XGBOOST_DIR / "nonrecursive_historical_regularization_study.csv"
MONTHLY_STABILITY_PATH = XGBOOST_DIR / "nonrecursive_historical_monthly_stability.csv"
RESIDUAL_DIAGNOSTICS_PATH = FIGURES_DIR / "nonrecursive_historical_residual_diagnostics.png"

FULLY_BLIND_METRICS_PATH = XGBOOST_DIR / "table8_fully_blind_population_weighted_metrics.json"
FULLY_BLIND_DIAGNOSTICS_PATH = XGBOOST_DIR / "table8_fully_blind_population_weighted_overfitting_diagnostics.json"
RECURSIVE_METRICS_PATH = XGBOOST_DIR / "table8_recursive_population_weighted_metrics.json"
ONE_STEP_METRICS_PATH = XGBOOST_DIR / "table8_sarima_guided_population_weighted_metrics.json"
ONE_STEP_TRAIN_METRICS_PATH = XGBOOST_DIR / "table8_sarima_guided_population_weighted_train_metrics.json"
FINAL_CANDIDATE_METRICS_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_metrics.json"

FINAL_CANDIDATE_LABEL = "Table 8 Non-Recursive Historical Population-Weighted"

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 525,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.25,
    "reg_lambda": 2.2,
    "random_state": 42,
    "n_jobs": 4,
    "tree_method": "hist",
}

REGULARIZATION_CONFIGS = [
    {
        "variant_key": "baseline",
        "label": "Baseline current parameters",
        "params_override": {},
    },
    {
        "variant_key": "model_a_lower_max_depth",
        "label": "Model A lower max_depth",
        "params_override": {"max_depth": 3},
    },
    {
        "variant_key": "model_b_higher_min_child_weight",
        "label": "Model B higher min_child_weight",
        "params_override": {"min_child_weight": 16},
    },
    {
        "variant_key": "model_c_lower_subsample_colsample",
        "label": "Model C lower subsample and colsample_bytree",
        "params_override": {"subsample": 0.65, "colsample_bytree": 0.6},
    },
    {
        "variant_key": "model_d_stronger_regularization",
        "label": "Model D stronger reg_alpha and reg_lambda",
        "params_override": {"reg_alpha": 0.75, "reg_lambda": 4.0},
    },
    {
        "variant_key": "model_e_combined_conservative",
        "label": "Model E combined conservative specification",
        "params_override": {
            "max_depth": 3,
            "min_child_weight": 18,
            "subsample": 0.65,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.9,
            "reg_lambda": 5.0,
        },
    },
]


def build_final_candidate_dataset() -> pd.DataFrame:
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


def fit_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, object],
) -> dict[str, object]:
    model = XGBRegressor(**params)
    model.fit(train_df[feature_columns], train_df["sqrt_consumption"])

    train_raw = model.predict(train_df[feature_columns])
    test_raw = model.predict(test_df[feature_columns])
    train_pred = invert_predictions("square_to_consumption", train_raw)
    test_pred = invert_predictions("square_to_consumption", test_raw)

    y_train = np.square(train_df["sqrt_consumption"].to_numpy())
    y_test = np.square(test_df["sqrt_consumption"].to_numpy())

    train_metrics = evaluate_predictions(y_train, train_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    output_df = pd.DataFrame(
        {
            "datetime": test_df["datetime"],
            "actual_consumption": y_test,
            "predicted_consumption": test_pred,
        }
    )
    output_df["residual"] = output_df["actual_consumption"] - output_df["predicted_consumption"]

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "test_output": output_df,
        "feature_importances": pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False),
    }


def build_gap_row(
    model_name: str,
    train_metrics: dict[str, float] | None,
    test_metrics: dict[str, float],
    note: str,
) -> dict[str, object]:
    row = {
        "model_name": model_name,
        "train_R2": np.nan,
        "train_MAE": np.nan,
        "train_RMSE": np.nan,
        "train_MAPE": np.nan,
        "test_R2": float(test_metrics["R2"]),
        "test_MAE": float(test_metrics["MAE"]),
        "test_RMSE": float(test_metrics["RMSE"]),
        "test_MAPE": float(test_metrics["MAPE"]),
        "gap_R2": np.nan,
        "gap_RMSE": np.nan,
        "gap_MAPE": np.nan,
        "note": note,
    }
    if train_metrics is not None:
        row.update(
            {
                "train_R2": float(train_metrics["R2"]),
                "train_MAE": float(train_metrics["MAE"]),
                "train_RMSE": float(train_metrics["RMSE"]),
                "train_MAPE": float(train_metrics["MAPE"]),
                "gap_R2": float(train_metrics["R2"] - test_metrics["R2"]),
                "gap_RMSE": float(test_metrics["RMSE"] - train_metrics["RMSE"]),
                "gap_MAPE": float(test_metrics["MAPE"] - train_metrics["MAPE"]),
            }
        )
    return row


def load_benchmark_rows() -> list[dict[str, object]]:
    fully_blind_test = json.loads(FULLY_BLIND_METRICS_PATH.read_text(encoding="utf-8"))
    fully_blind_diag = json.loads(FULLY_BLIND_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    recursive_test = json.loads(RECURSIVE_METRICS_PATH.read_text(encoding="utf-8"))
    one_step_test = json.loads(ONE_STEP_METRICS_PATH.read_text(encoding="utf-8"))
    one_step_train = json.loads(ONE_STEP_TRAIN_METRICS_PATH.read_text(encoding="utf-8"))

    return [
        build_gap_row(
            model_name="Official Fully Blind Population-Weighted",
            train_metrics=fully_blind_diag["train_metrics"],
            test_metrics=fully_blind_diag["test_metrics"],
            note="Official 2022-2024 direct-batch benchmark.",
        ),
        build_gap_row(
            model_name="Table 8 Recursive Population-Weighted",
            train_metrics=one_step_train["train_metrics_consumption_scale"],
            test_metrics=recursive_test["test_metrics_consumption_scale"],
            note=(
                "Train metrics come from the shared SARIMA-guided fitted model; "
                "test metrics come from recursive rollout, so gap is conservative but not perfectly apples-to-apples."
            ),
        ),
        build_gap_row(
            model_name="Table 8 SARIMA-Guided One-Step-Ahead Population-Weighted",
            train_metrics=one_step_train["train_metrics_consumption_scale"],
            test_metrics=one_step_test["test_metrics_consumption_scale"],
            note="Uses observed past 2025 consumption via one-step-ahead target-derived lag features.",
        ),
    ]


def compute_monthly_stability(test_output: pd.DataFrame) -> pd.DataFrame:
    monthly_rows: list[dict[str, object]] = []
    test_output = test_output.copy()
    test_output["month_period"] = pd.to_datetime(test_output["datetime"]).dt.to_period("M")

    for month_period, month_df in test_output.groupby("month_period"):
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
    monthly_df["rmse_change_vs_prev_month"] = monthly_df["monthly_RMSE"].diff()
    monthly_df["mape_change_vs_prev_month"] = monthly_df["monthly_MAPE"].diff()
    return monthly_df


def plot_residual_diagnostics(test_output: pd.DataFrame) -> None:
    output_df = test_output.copy()
    output_df["datetime"] = pd.to_datetime(output_df["datetime"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    axes[0].plot(output_df["datetime"], output_df["residual"], linewidth=0.9)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residual vs Time")
    axes[0].set_xlabel("Datetime")
    axes[0].set_ylabel("Residual")

    axes[1].hist(output_df["residual"], bins=50, edgecolor="black", alpha=0.8)
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    axes[2].scatter(output_df["predicted_consumption"], output_df["residual"], s=6, alpha=0.25)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Residual vs Prediction")
    axes[2].set_xlabel("Predicted Consumption")
    axes[2].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig(RESIDUAL_DIAGNOSTICS_PATH, dpi=160)
    plt.close()


def main() -> None:
    validate_dependencies_and_files()

    dataset = build_final_candidate_dataset()
    train_df, test_df = split_train_test(dataset)

    final_result = fit_variant(
        train_df=train_df,
        test_df=test_df,
        feature_columns=EXPERIMENTAL_FEATURES,
        params=dict(BASE_PARAMS),
    )

    final_row = build_gap_row(
        model_name=FINAL_CANDIDATE_LABEL,
        train_metrics=final_result["train_metrics"],
        test_metrics=final_result["test_metrics"],
        note=(
            "Final direct-batch candidate using baseline + cyclic + historical analog features. "
            "Effective train window is 2023-2024 because analog features require previous-year history."
        ),
    )

    benchmark_rows = [final_row, *load_benchmark_rows()]
    overfitting_df = pd.DataFrame(benchmark_rows)
    overfitting_df.to_csv(OVERFITTING_AUDIT_PATH, index=False)

    regularization_rows: list[dict[str, object]] = []
    for config in REGULARIZATION_CONFIGS:
        params = dict(BASE_PARAMS)
        params.update(config["params_override"])
        result = fit_variant(
            train_df=train_df,
            test_df=test_df,
            feature_columns=EXPERIMENTAL_FEATURES,
            params=params,
        )
        regularization_rows.append(
            {
                "variant_key": config["variant_key"],
                "label": config["label"],
                "params": json.dumps(params, ensure_ascii=False),
                "train_R2": float(result["train_metrics"]["R2"]),
                "train_MAE": float(result["train_metrics"]["MAE"]),
                "train_RMSE": float(result["train_metrics"]["RMSE"]),
                "train_MAPE": float(result["train_metrics"]["MAPE"]),
                "test_R2": float(result["test_metrics"]["R2"]),
                "test_MAE": float(result["test_metrics"]["MAE"]),
                "test_RMSE": float(result["test_metrics"]["RMSE"]),
                "test_MAPE": float(result["test_metrics"]["MAPE"]),
                "gap_R2": float(result["train_metrics"]["R2"] - result["test_metrics"]["R2"]),
                "gap_RMSE": float(result["test_metrics"]["RMSE"] - result["train_metrics"]["RMSE"]),
                "gap_MAPE": float(result["test_metrics"]["MAPE"] - result["train_metrics"]["MAPE"]),
            }
        )

    regularization_df = pd.DataFrame(regularization_rows)
    regularization_df.to_csv(REGULARIZATION_STUDY_PATH, index=False)

    monthly_df = compute_monthly_stability(final_result["test_output"])
    monthly_df.to_csv(MONTHLY_STABILITY_PATH, index=False)

    plot_residual_diagnostics(final_result["test_output"])

    print("Overfitting / robustness audit complete.")
    print()
    print("Step 1: Train/Test Generalization Audit")
    print(overfitting_df.to_string(index=False))
    print()
    print("Step 2: Regularization Robustness Study")
    print(regularization_df.to_string(index=False))
    print()
    print("Step 3: Monthly Stability")
    print(monthly_df.to_string(index=False))
    print()
    print(f"Saved: {OVERFITTING_AUDIT_PATH}")
    print(f"Saved: {REGULARIZATION_STUDY_PATH}")
    print(f"Saved: {MONTHLY_STABILITY_PATH}")
    print(f"Saved: {RESIDUAL_DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
