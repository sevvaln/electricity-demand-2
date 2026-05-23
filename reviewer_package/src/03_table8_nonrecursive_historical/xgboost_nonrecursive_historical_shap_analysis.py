from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from xgboost_population_weighted_nonrecursive_historical_experiment import (
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
METRICS_PATH = XGBOOST_DIR / "table8_nonrecursive_historical_population_weighted_metrics.json"
SUMMARY_PLOT_PATH = FIGURES_DIR / "shap_summary_final_model.png"
BAR_PLOT_PATH = FIGURES_DIR / "shap_bar_final_model.png"
MEAN_ABS_CSV_PATH = XGBOOST_DIR / "shap_values_mean_abs_final_model.csv"
DEPENDENCE_CDD_PATH = FIGURES_DIR / "shap_dependence_weighted_CDD.png"
DEPENDENCE_HDD_PATH = FIGURES_DIR / "shap_dependence_weighted_HDD.png"
DEPENDENCE_HISTORY_PATH = FIGURES_DIR / "shap_dependence_historical_profile.png"
DEPENDENCE_CALENDAR_PATH = FIGURES_DIR / "shap_dependence_calendar_effects.png"
INTERPRETATION_PATH = XGBOOST_DIR / "shap_final_model_interpretation.txt"

RANDOM_STATE = 42


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_final_dataset(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_df, _ = build_population_weighted_base_dataset(TRAIN_YEARS + [TEST_YEAR])
    augmented_df = add_nonrecursive_historical_features(base_df)
    selected_columns = ["datetime", "year", "consumption", "sqrt_consumption", *feature_columns]
    dataset = augmented_df[selected_columns].copy()
    dataset = dataset.apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name != "datetime"
        else column
    )
    dataset = dataset.dropna().sort_values("datetime").reset_index(drop=True)
    train_df, test_df = split_train_test(dataset)
    return train_df, test_df


def fit_final_model(train_df: pd.DataFrame, feature_columns: list[str], params: dict[str, object]) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.fit(train_df[feature_columns], train_df["sqrt_consumption"])
    return model


def validate_reproduced_metrics(
    model: XGBRegressor,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    expected_test_metrics: dict[str, float],
) -> dict[str, float]:
    raw_test = model.predict(test_df[feature_columns])
    test_pred = invert_predictions("square_to_consumption", raw_test)
    y_test = np.square(test_df["sqrt_consumption"].to_numpy())
    reproduced = evaluate_predictions(y_test, test_pred)

    for metric_name, expected_value in expected_test_metrics.items():
        if abs(reproduced[metric_name] - float(expected_value)) > 1e-6:
            raise ValueError(
                f"Final model reproduction check failed for {metric_name}: "
                f"expected {expected_value}, got {reproduced[metric_name]}"
            )
    return reproduced


def compute_shap_values(model: XGBRegressor, X_test: pd.DataFrame) -> tuple[np.ndarray, float]:
    dmatrix = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contributions[:, :-1]
    base_value = float(np.mean(contributions[:, -1]))
    return shap_values, base_value


def plot_summary(shap_values: np.ndarray, X_test: pd.DataFrame, mean_abs_df: pd.DataFrame) -> None:
    top_features = mean_abs_df["feature"].tolist()[:15]
    X_plot = X_test[top_features]
    feature_to_index = {feature: idx for idx, feature in enumerate(X_test.columns)}

    plt.figure(figsize=(12, 8))
    cmap = plt.cm.coolwarm
    for y_index, feature in enumerate(reversed(top_features), start=1):
        column_index = feature_to_index[feature]
        feature_values = X_plot[feature].to_numpy()
        shap_column = shap_values[:, column_index]

        if np.nanmax(feature_values) > np.nanmin(feature_values):
            color_values = (feature_values - np.nanmin(feature_values)) / (
                np.nanmax(feature_values) - np.nanmin(feature_values)
            )
        else:
            color_values = np.full_like(feature_values, 0.5, dtype=float)

        rng = np.random.default_rng(RANDOM_STATE + y_index)
        jitter = rng.normal(loc=0.0, scale=0.12, size=len(feature_values))
        plt.scatter(
            shap_column,
            np.full(len(feature_values), y_index) + jitter,
            c=color_values,
            cmap=cmap,
            s=10,
            alpha=0.35,
            edgecolors="none",
        )

    plt.yticks(range(1, len(top_features) + 1), list(reversed(top_features)))
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("SHAP value contribution to sqrt_consumption prediction")
    plt.ylabel("Feature")
    plt.title("SHAP Summary Plot: Final Table 8 Non-Recursive Historical Model (2025)")
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=160)
    plt.close()


def plot_bar(mean_abs_df: pd.DataFrame) -> None:
    top_df = mean_abs_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("SHAP Bar Importance: Final Table 8 Non-Recursive Historical Model (2025)")
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH, dpi=160)
    plt.close()


def single_dependence_plot(
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    feature_name: str,
    output_path: Path,
) -> None:
    feature_index = list(X_test.columns).index(feature_name)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_test[feature_name],
        shap_values[:, feature_index],
        s=10,
        alpha=0.35,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel(feature_name)
    plt.ylabel("SHAP value")
    plt.title(f"SHAP Dependence: {feature_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def grouped_dependence_plot(
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(feature_names), figsize=(7 * len(feature_names), 5))
    if len(feature_names) == 1:
        axes = [axes]

    for ax, feature_name in zip(axes, feature_names):
        feature_index = list(X_test.columns).index(feature_name)
        ax.scatter(
            X_test[feature_name],
            shap_values[:, feature_index],
            s=10,
            alpha=0.35,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("SHAP value")
        ax.set_title(feature_name)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_interpretation(mean_abs_df: pd.DataFrame) -> str:
    top10 = mean_abs_df.head(10).copy()
    top_features = top10["feature"].tolist()
    historical_features = {
        "prev_year_same_month_day_hour_sqrt",
        "prev_year_same_month_day_prev_hour_sqrt",
        "prev_year_same_month_day_seasonal_diff_24h_sqrt",
        "prev_year_mean_sqrt_by_hour_day_of_week",
        "prev_year_mean_sqrt_by_hour_month",
    }
    weather_features = {"weighted_HDD", "weighted_CDD"}
    calendar_features = {"is_public_holiday", "weekend", "night", "is_holiday_window", "is_bridge_day"}

    historical_in_top10 = [feature for feature in top_features if feature in historical_features]
    weather_in_top10 = [feature for feature in top_features if feature in weather_features]
    calendar_in_top10 = [feature for feature in top_features if feature in calendar_features]

    lines = [
        "SHAP Interpretation: Final Table 8 Non-Recursive Historical Population-Weighted Forecast",
        "",
        f"Most influential features by mean absolute SHAP value: {', '.join(top_features[:6])}.",
        (
            "Historical analog features dominate the ranking."
            if historical_in_top10
            else "Historical analog features do not dominate the top SHAP ranking."
        ),
        (
            f"Historical analog features appearing in the top 10: {', '.join(historical_in_top10)}."
            if historical_in_top10
            else "No historical analog feature appears in the top 10."
        ),
        (
            f"Weather variables still matter: {', '.join(weather_in_top10)} appear among the top drivers."
            if weather_in_top10
            else "Weather variables are not among the strongest SHAP drivers in the top 10, though they may still contribute."
        ),
        (
            f"Calendar and holiday effects remain meaningful through: {', '.join(calendar_in_top10)}."
            if calendar_in_top10
            else "Calendar and holiday indicators are present but not dominant in the top SHAP ranking."
        ),
        (
            "Interpretation is on the sqrt_consumption prediction scale, because the final model is trained on sqrt_consumption."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    validate_dependencies_and_files()
    metrics_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    feature_columns = list(metrics_payload["feature_columns"])
    params = dict(metrics_payload["hyperparameters"])

    train_df, test_df = build_final_dataset(feature_columns)
    model = fit_final_model(train_df, feature_columns, params)
    validate_reproduced_metrics(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
        expected_test_metrics=metrics_payload["test_metrics"],
    )

    X_test = test_df[feature_columns].copy()
    shap_values, base_value = compute_shap_values(model, X_test)

    mean_abs_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    mean_abs_df["base_value"] = base_value
    mean_abs_df.to_csv(MEAN_ABS_CSV_PATH, index=False)

    plot_summary(shap_values=shap_values, X_test=X_test, mean_abs_df=mean_abs_df)
    plot_bar(mean_abs_df)
    single_dependence_plot(X_test, shap_values, "weighted_CDD", DEPENDENCE_CDD_PATH)
    single_dependence_plot(X_test, shap_values, "weighted_HDD", DEPENDENCE_HDD_PATH)
    grouped_dependence_plot(
        X_test,
        shap_values,
        ["prev_year_mean_sqrt_by_hour_day_of_week", "prev_year_mean_sqrt_by_hour_month"],
        DEPENDENCE_HISTORY_PATH,
        "SHAP Dependence: Historical Profile Features",
    )
    grouped_dependence_plot(
        X_test,
        shap_values,
        ["is_public_holiday", "weekend"],
        DEPENDENCE_CALENDAR_PATH,
        "SHAP Dependence: Calendar and Holiday Effects",
    )

    interpretation = build_interpretation(mean_abs_df)
    save_text(INTERPRETATION_PATH, interpretation)

    print("Final model SHAP analysis complete.")
    print()
    print(mean_abs_df.head(15).to_string(index=False))
    print()
    print(interpretation)


if __name__ == "__main__":
    main()
