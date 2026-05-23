from __future__ import annotations

from pathlib import Path

import pandas as pd

from xgboost_population_weighted_pipeline import MODEL_CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parent
XGBOOST_DIR = PROJECT_ROOT / "outputs" / "xgboost"

OUTPUT_CSV_PATH = XGBOOST_DIR / "xgboost_population_weighted_spec_consistency.csv"
OUTPUT_TXT_PATH = XGBOOST_DIR / "xgboost_population_weighted_spec_consistency.txt"

INTENDED_SPECS = {
    "table3_intramonth_population_weighted": {
        "model_name": "Table 3 Population-Weighted",
        "intended_target": "consumption",
        "intended_features": [
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
        ],
        "allow_extra_ml_adaptation": False,
        "notes": "Original Table 3 econometric specification.",
    },
    "table3_sarima_guided_population_weighted": {
        "model_name": "Table 3 Population-Weighted SARIMA-Guided",
        "intended_target": "consumption",
        "intended_features": [
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
        ],
        "allowed_extra_features": [
            "log_consumption_lag_1h",
            "log_consumption_seasonal_diff_24h",
        ],
        "allow_extra_ml_adaptation": True,
        "notes": "SARIMA-guided ML adaptation is allowed on top of Table 3 base specification.",
    },
    "table8_sarima_guided_population_weighted": {
        "model_name": "Table 8 Population-Weighted SARIMA-Guided",
        "intended_target": "sqrt_consumption",
        "intended_features": [
            "weighted_HDD",
            "weighted_CDD",
            "night",
            "weekend",
            "PMI_prev_month",
            "IR_prev_month",
        ],
        "allowed_extra_features": [
            "log_consumption_lag_1h",
            "log_consumption_seasonal_diff_24h",
        ],
        "allow_extra_ml_adaptation": True,
        "notes": "Original Table 8 excludes CUR; SARIMA-guided lag features are allowed as ML adaptation.",
    },
    "table8_fully_blind_population_weighted": {
        "model_name": "Table 8 Fully Blind Broad Future Forecast",
        "intended_target": "sqrt_consumption",
        "intended_features": [
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
        "forbidden_features": [
            "CUR_prev_month",
            "log_consumption_lag_1h",
            "log_consumption_seasonal_diff_24h",
            "consumption_lag_1h",
            "consumption_lag_24h",
            "consumption_lag_168h",
            "consumption_roll_mean_24h",
            "consumption_roll_std_24h",
            "consumption_roll_mean_168h",
            "consumption_roll_std_168h",
        ],
        "allow_extra_ml_adaptation": False,
        "notes": "Original Table 8 excludes CUR. Fully blind extension may add only calendar/holiday features, not target-derived features.",
    },
}


def stringify_features(values: list[str]) -> str:
    return " | ".join(values)


def build_consistency_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for config in MODEL_CONFIGS:
        model_key = str(config["model_key"])
        if model_key not in INTENDED_SPECS:
            continue

        intended = INTENDED_SPECS[model_key]
        actual_target = str(config["target_column"])
        actual_features = list(config["feature_columns"])
        intended_target = str(intended["intended_target"])
        intended_features = list(intended["intended_features"])
        allowed_extra = list(intended.get("allowed_extra_features", []))
        forbidden_features = list(intended.get("forbidden_features", []))

        target_match = actual_target == intended_target
        missing_from_actual = [feature for feature in intended_features if feature not in actual_features]
        unexpected_features = [
            feature
            for feature in actual_features
            if feature not in intended_features and feature not in allowed_extra
        ]
        forbidden_in_actual = [feature for feature in actual_features if feature in forbidden_features]

        feature_match = not missing_from_actual and not unexpected_features and not forbidden_in_actual
        note_parts = [str(intended["notes"])]
        if missing_from_actual:
            note_parts.append(f"Missing intended features: {missing_from_actual}")
        if unexpected_features:
            note_parts.append(f"Unexpected features: {unexpected_features}")
        if forbidden_in_actual:
            note_parts.append(f"Forbidden features present: {forbidden_in_actual}")
        if model_key == "table8_sarima_guided_population_weighted" and "CUR_prev_month" not in actual_features:
            note_parts.append("CUR correctly excluded.")
        if model_key == "table8_fully_blind_population_weighted" and "CUR_prev_month" not in actual_features:
            note_parts.append("CUR correctly excluded from fully blind model.")

        rows.append(
            {
                "model_name": intended["model_name"],
                "intended_target": intended_target,
                "actual_target": actual_target,
                "target_match": target_match,
                "intended_features": stringify_features(intended_features + allowed_extra),
                "actual_features": stringify_features(actual_features),
                "feature_match": feature_match,
                "notes": " ".join(note_parts),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(consistency_df: pd.DataFrame) -> None:
    consistency_df.to_csv(OUTPUT_CSV_PATH, index=False)

    lines = [
        "XGBoost Population-Weighted Specification Consistency Check",
        "",
        consistency_df.to_string(index=False),
    ]
    OUTPUT_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    consistency_df = build_consistency_table()
    write_outputs(consistency_df)
    print(consistency_df.to_string(index=False))


if __name__ == "__main__":
    main()
