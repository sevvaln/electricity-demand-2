from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from itertools import combinations
from pathlib import Path
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

try:
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.metrics import r2_score
except Exception as exc:  # pragma: no cover - user environment guard
    LassoCV = None
    RidgeCV = None
    ElasticNetCV = None
    r2_score = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TARGET_YEARS = [2022, 2023, 2024]
BASE_HDD_THRESHOLD = 18.0
BASE_CDD_THRESHOLD = 24.0
ALL_CONTINUOUS_COLUMNS = ["HDD", "CDD", "PMI", "IR", "CUR"]
BINARY_COLUMNS = ["weekend", "night"]
LASSO_RANDOM_STATE = 42
LASSO_MAX_ITER = 20000
LASSO_SELECTION_TOLERANCE = 1e-6
P_VALUE_SELECTION_THRESHOLD = 0.05
RIDGE_ALPHA_GRID = np.logspace(-4, 4, 60)
ELASTIC_NET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
POLYNOMIAL_SOURCE_COLUMNS = {"HDD", "CDD", "PMI", "IR", "CUR", "hour", "month"}
MONTH_LEVEL_COLUMNS = {"month", "PMI", "IR", "CUR"}
FORCED_INTERACTION_PAIRS = [
    ("HDD", "weekend"),
    ("CDD", "hour"),
    ("CDD", "month"),
    ("HDD", "night"),
    ("PMI", "HDD"),
    ("CUR", "CDD"),
    ("IR", "PMI"),
]
STRUCTURALLY_INVALID_INTERACTION_PAIRS = {
    frozenset({"HDD", "CDD"}),
}
DESIGN_STD_TOLERANCE = 1e-10
DESIGN_RANK_TOLERANCE = 1e-10

CONSUMPTION_FILES = {
    2022: RAW_DATA_ROOT / "consumption" / "tuketim_2022.xlsx",
    2023: RAW_DATA_ROOT / "consumption" / "tuketim_2023.xlsx",
    2024: RAW_DATA_ROOT / "consumption" / "tuketim_2024.xlsx",
    2025: RAW_DATA_ROOT / "consumption" / "tuketim_2025.xlsx",
}

TEMPERATURE_FILES = {
    2022: RAW_DATA_ROOT / "temperature" / "sicaklik_2022.xlsx",
    2023: RAW_DATA_ROOT / "temperature" / "sicaklik_2023.xlsx",
    2024: RAW_DATA_ROOT / "temperature" / "sicaklik_2024.xlsx",
    2025: RAW_DATA_ROOT / "temperature" / "sicaklik_2025.xlsx",
}

ECONOMIC_FILES = {
    "PMI": RAW_DATA_ROOT / "economic" / "pmi_degerleri.xlsx",
    "IR": RAW_DATA_ROOT / "economic" / "faiz_oranlari.xlsx",
    "CUR": RAW_DATA_ROOT / "economic" / "kapasite_kullanim_orani.xlsx",
}

TURKISH_CHAR_MAP = str.maketrans(
    {
        "\u00e7": "c",
        "\u00c7": "C",
        "\u011f": "g",
        "\u011e": "G",
        "\u0131": "i",
        "\u0130": "I",
        "\u00f6": "o",
        "\u00d6": "O",
        "\u015f": "s",
        "\u015e": "S",
        "\u00fc": "u",
        "\u00dc": "U",
    }
)

DEFAULT_CITY_NAME_MAPPING = {
    "KARABUK": "KARABÜK",
    "KARABÜK KAPULLU": "KARABÜK",
}


@dataclass(frozen=True)
class ZScoreScaler:
    means: dict[str, float]
    stds: dict[str, float]
    columns: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.columns:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce") - self.means[column]
            ) / self.stds[column]
        return out


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _normalize_label(value: object) -> str:
    text = str(value).translate(TURKISH_CHAR_MAP)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def standardize_city_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u200b", " ")
        .replace("\u200c", " ")
        .replace("\u200d", " ")
        .replace("\ufeff", " ")
        .replace("\xa0", " ")
    )
    text = " ".join(text.split()).strip().upper()
    return text


def build_city_match_key(value: object) -> str:
    text = standardize_city_name(value)
    text = text.translate(TURKISH_CHAR_MAP)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split()).strip()


def parse_turkish_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = unicodedata.normalize("NFKC", str(value))
    text = (
        text.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\xa0", "")
        .replace(" ", "")
    )
    if text in {"", "nan", "NaN", "None"}:
        return float("nan")

    text = text.replace(".", "").replace(",", ".")
    return float(pd.to_numeric(text, errors="coerce"))


def _rename_by_alias(
    df: pd.DataFrame,
    alias_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    normalized = {_normalize_label(column): column for column in df.columns}
    rename_map: dict[str, str] = {}
    for target, aliases in alias_map.items():
        for alias in aliases:
            original = normalized.get(alias)
            if original is not None:
                rename_map[original] = target
                break
    return df.rename(columns=rename_map)


def _excel_date_to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    as_object = series.astype("object")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    python_date_mask = as_object.map(
        lambda value: isinstance(value, (datetime, date, pd.Timestamp))
    )
    if python_date_mask.any():
        result.loc[python_date_mask] = pd.to_datetime(
            as_object.loc[python_date_mask], errors="coerce"
        )

    remaining = ~python_date_mask
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
            )

        text_mask = ~numeric_mask
        if text_mask.any():
            result.loc[text_mask.index[text_mask]] = pd.to_datetime(
                as_object.loc[text_mask.index[text_mask]],
                errors="coerce",
                dayfirst=True,
            )
    return result


def _excel_time_to_string(series: pd.Series) -> pd.Series:
    as_object = series.astype("object")
    result = pd.Series("", index=series.index, dtype="object")

    datetime_mask = as_object.map(
        lambda value: isinstance(value, (datetime, pd.Timestamp))
    )
    if datetime_mask.any():
        parsed = pd.to_datetime(as_object.loc[datetime_mask], errors="coerce")
        result.loc[datetime_mask] = parsed.dt.strftime("%H:%M:%S")

    pure_time_mask = as_object.map(lambda value: isinstance(value, time))
    if pure_time_mask.any():
        result.loc[pure_time_mask] = as_object.loc[pure_time_mask].map(
            lambda value: value.strftime("%H:%M:%S")
        )

    remaining = ~(datetime_mask | pure_time_mask)
    if remaining.any():
        numeric = pd.to_numeric(as_object.loc[remaining], errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask.index[numeric_mask]] = (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(numeric.loc[numeric_mask], unit="D")
            ).strftime("%H:%M:%S")

        text_index = numeric_mask.index[~numeric_mask]
        if len(text_index) > 0:
            text = as_object.loc[text_index].astype(str).str.strip()
            text = text.replace({"nan": "", "NaT": "", "None": ""})

            hours_only = text.str.match(r"^\d{1,2}$", na=False)
            text = text.where(~hours_only, text.str.zfill(2) + ":00:00")

            hhmm = text.str.match(r"^\d{1,2}:\d{2}$", na=False)
            text = text.where(
                ~hhmm,
                text.str.replace(
                    r"^(\d{1,2}:\d{2})$",
                    lambda match: match.group(1).zfill(5) + ":00",
                    regex=True,
                ),
            )

            hhmmss = text.str.match(r"^\d{1,2}:\d{2}:\d{2}$", na=False)
            text = text.where(
                ~hhmmss,
                text.str.replace(
                    r"^(\d{1,2}):",
                    lambda match: match.group(1).zfill(2) + ":",
                    regex=True,
                ),
            )
            result.loc[text_index] = text
    return result


def _build_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = _excel_date_to_datetime(date_series)
    times = _excel_time_to_string(time_series)
    return pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times,
        errors="coerce",
    )


def _read_excel(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", **kwargs)


def load_consumption(year: int) -> pd.DataFrame:
    path = CONSUMPTION_FILES.get(year)
    if path is None:
        raise KeyError(f"{year} icin tuketim dosya yolu tanimli degil.")
    if not path.exists():
        raise FileNotFoundError(f"Tuketim dosyasi bulunamadi: {path}")
    df = _read_excel(path)
    df = _rename_by_alias(
        df,
        {
            "date": ("tarih",),
            "time": ("saat",),
            "consumption": ("tuketimmiktarimwh", "consumption"),
        },
    )
    if {"date", "time", "consumption"} - set(df.columns):
        df = df.rename(
            columns={
                df.columns[0]: "date",
                df.columns[1]: "time",
                df.columns[2]: "consumption",
            }
        )
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    return (
        df[["datetime", "consumption"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_temperature(year: int) -> pd.DataFrame:
    path = TEMPERATURE_FILES.get(year)
    if path is None:
        raise KeyError(f"{year} icin sicaklik dosya yolu tanimli degil.")
    if not path.exists():
        raise FileNotFoundError(f"Sicaklik dosyasi bulunamadi: {path}")
    df = _read_excel(path)
    df = _rename_by_alias(
        df,
        {
            "city": ("istasyonadi", "city"),
            "date": ("tarih", "date"),
            "time": ("saat", "time"),
            "temperature": ("sicaklik", "temperature"),
        },
    )
    if {"city", "date", "time", "temperature"} - set(df.columns):
        df = df.rename(
            columns={
                df.columns[1]: "city",
                df.columns[3]: "date",
                df.columns[4]: "time",
                df.columns[5]: "temperature",
            }
        )
    df["datetime"] = _build_datetime(df["date"], df["time"])
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    return (
        df[["datetime", "city", "temperature"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def load_economic_data() -> pd.DataFrame:
    pmi = _read_excel(ECONOMIC_FILES["PMI"])
    pmi = _rename_by_alias(pmi, {"date": ("date", "tarih"), "PMI": ("pmi",)})
    pmi["date"] = _excel_date_to_datetime(pmi["date"])
    pmi["PMI"] = pd.to_numeric(pmi["PMI"], errors="coerce")
    pmi["year_month"] = pmi["date"].dt.to_period("M")
    pmi = pmi[["year_month", "PMI"]]

    interest = _read_excel(ECONOMIC_FILES["IR"], header=None, names=["date", "IR"])
    interest["date"] = _excel_date_to_datetime(interest["date"])
    interest["IR"] = pd.to_numeric(interest["IR"], errors="coerce")
    interest["year_month"] = interest["date"].dt.to_period("M")
    interest = interest[["year_month", "IR"]]

    capacity = _read_excel(ECONOMIC_FILES["CUR"])
    capacity = _rename_by_alias(
        capacity,
        {
            "date": ("date", "tarih"),
            "CUR": ("cur", "kko"),
        },
    )
    capacity["date"] = _excel_date_to_datetime(capacity["date"])
    capacity["CUR"] = pd.to_numeric(
        capacity["CUR"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    capacity["year_month"] = capacity["date"].dt.to_period("M")
    capacity = capacity[["year_month", "CUR"]]

    econ = pmi.merge(interest, on="year_month", how="outer")
    econ = econ.merge(capacity, on="year_month", how="outer")
    econ = econ.sort_values("year_month").drop_duplicates("year_month")
    return econ.reset_index(drop=True)


def build_analysis_frame(years: list[int] | tuple[int, ...] | None = None) -> pd.DataFrame:
    selected_years = list(years) if years is not None else TARGET_YEARS
    consumption = pd.concat(
        [load_consumption(year) for year in selected_years],
        ignore_index=True,
    )
    temperature = pd.concat(
        [load_temperature(year) for year in selected_years],
        ignore_index=True,
    )
    temperature_hourly = (
        temperature.groupby("datetime", as_index=False)["temperature"].mean()
    )

    df = consumption.merge(temperature_hourly, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)

    df["HDD"] = np.maximum(BASE_HDD_THRESHOLD - df["temperature"], 0)
    df["CDD"] = np.maximum(df["temperature"] - BASE_CDD_THRESHOLD, 0)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["year_month"] = df["datetime"].dt.to_period("M")
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["night"] = df["hour"].between(0, 6).astype(int)
    df["log_consumption"] = np.log(df["consumption"])
    df["sqrt_consumption"] = np.sqrt(df["consumption"])

    econ = load_economic_data()
    df = df.merge(econ, on="year_month", how="left")
    return df


def load_population_data(population_path: Path) -> pd.DataFrame:
    if not population_path.exists():
        raise FileNotFoundError(f"Nufus dosyasi bulunamadi: {population_path}")

    population = _read_excel(population_path)
    population = _rename_by_alias(
        population,
        {
            "city": ("istasyonadi", "il", "sehir", "province", "city"),
            "population": ("nufus", "population", "toplamnufus"),
        },
    )

    required = {"city", "population"}
    if required - set(population.columns):
        population = population.rename(
            columns={
                population.columns[0]: "city",
                population.columns[1]: "population",
            }
        )

    population = population[["city", "population"]].copy()
    population["city_original"] = population["city"].astype(str)
    population["city_standardized"] = population["city_original"].map(standardize_city_name)
    population["city_key"] = population["city_standardized"].map(build_city_match_key)
    population["population_raw"] = population["population"]
    population["population"] = population["population_raw"].map(parse_turkish_numeric)
    return population


def build_population_weighted_degree_hour_frame(
    years: list[int] | tuple[int, ...],
    population_path: Path,
    output_dir: Path | None = None,
    manual_city_mapping: dict[str, str] | None = None,
    require_full_target_coverage: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if output_dir is None:
        output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    population = load_population_data(population_path)
    mapping_source = manual_city_mapping or DEFAULT_CITY_NAME_MAPPING
    normalized_mapping = {
        build_city_match_key(source): build_city_match_key(target)
        for source, target in mapping_source.items()
    }

    temperature = pd.concat(
        [load_temperature(year).assign(source_year=year) for year in years],
        ignore_index=True,
    )
    temperature["city_original"] = temperature["city"].astype(str)
    temperature["city_standardized"] = temperature["city_original"].map(standardize_city_name)
    temperature["city_key_raw"] = temperature["city_standardized"].map(build_city_match_key)
    temperature["city_key"] = temperature["city_key_raw"].map(
        lambda value: normalized_mapping.get(value, value)
    )

    unique_temperature_cities = (
        temperature[["city_original", "city_standardized", "city_key_raw", "city_key"]]
        .drop_duplicates()
        .sort_values(["city_key", "city_original"])
        .reset_index(drop=True)
    )
    unique_population_cities = (
        population[["city_original", "city_standardized", "city_key", "population"]]
        .drop_duplicates()
        .sort_values(["city_key", "city_original"])
        .reset_index(drop=True)
    )

    matched_cities = unique_temperature_cities.merge(
        unique_population_cities[["city_original", "city_key", "population"]].rename(
            columns={"city_original": "population_city_original"}
        ),
        on="city_key",
        how="inner",
    )
    unmatched_temperature = unique_temperature_cities.loc[
        ~unique_temperature_cities["city_key"].isin(unique_population_cities["city_key"])
    ].copy()
    unmatched_population = unique_population_cities.loc[
        ~unique_population_cities["city_key"].isin(unique_temperature_cities["city_key"])
    ].copy()

    unique_temperature_cities.to_csv(
        output_dir / "population_weighted_temperature_unique_cities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unique_population_cities.to_csv(
        output_dir / "population_weighted_population_unique_cities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matched_cities.to_csv(
        output_dir / "population_weighted_matched_cities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched_temperature.to_csv(
        output_dir / "population_weighted_unmatched_temperature_cities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched_population.to_csv(
        output_dir / "population_weighted_unmatched_population_cities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_summary_lines = [
        "Population / Temperature City Match Validation",
        "",
        f"Years: {', '.join(str(year) for year in years)}",
        f"Temperature unique raw cities: {unique_temperature_cities['city_original'].nunique()}",
        f"Population unique raw cities: {unique_population_cities['city_original'].nunique()}",
        f"Matched city count: {matched_cities['city_key'].nunique()}",
        f"Every temperature city has a population value: {unmatched_temperature.empty}",
        "",
        "Manual city-name mapping dictionary:",
    ]
    for source, target in mapping_source.items():
        match_summary_lines.append(f"- {source!r}: {target!r}")
    match_summary_lines.extend(
        [
            "",
            "Unmatched temperature cities:",
            "(none)" if unmatched_temperature.empty else unmatched_temperature.to_string(index=False),
            "",
            "Unmatched population cities:",
            "(none)" if unmatched_population.empty else unmatched_population.to_string(index=False),
            "",
            "Sample matched rows:",
            matched_cities.head(10).to_string(index=False),
        ]
    )
    (output_dir / "population_weighted_match_validation.txt").write_text(
        "\n".join(match_summary_lines),
        encoding="utf-8",
    )

    if not unmatched_temperature.empty or not unmatched_population.empty:
        raise ValueError(
            "Sehir eslestirmesi tamamlanamadi. Detaylar `population_weighted_match_validation.txt` dosyasina yazildi."
        )

    if population["population"].isna().any():
        invalid_population = population.loc[population["population"].isna(), ["city_original", "population_raw"]]
        invalid_population.to_csv(
            output_dir / "population_weighted_invalid_population_rows.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError("Nufus serisinde sayisal parse edilemeyen degerler var.")

    population_min = float(population["population"].min())
    population_max = float(population["population"].max())
    population_total = float(population["population"].sum())
    if population_min <= 0 or population_max > 30_000_000 or not (50_000_000 <= population_total <= 120_000_000):
        raise ValueError(
            "Nufus degerleri sayisal olarak parse edildi ancak makul gorunmuyor. Lutfen formatlamayi kontrol edin."
        )

    weights = unique_population_cities[["city_key", "city_original", "population"]].copy()
    weights = weights.drop_duplicates("city_key").sort_values("city_key").reset_index(drop=True)
    weights["weight"] = weights["population"] / weights["population"].sum()
    weights.to_csv(
        output_dir / "population_weighted_city_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weight_lines = [
        "Population Weight Validation",
        "",
        f"Population sum: {weights['population'].sum():,.0f}",
        f"Weight sum: {weights['weight'].sum():.12f}",
        f"Min weight: {weights['weight'].min():.12f}",
        f"Max weight: {weights['weight'].max():.12f}",
        "Weight sum approximately 1: "
        f"{bool(np.isclose(weights['weight'].sum(), 1.0, atol=1e-10))}",
        "",
        "Top 10 cities/provinces by weight:",
        weights.sort_values("weight", ascending=False).head(10).to_string(index=False),
    ]
    (output_dir / "population_weighted_weight_validation.txt").write_text(
        "\n".join(weight_lines),
        encoding="utf-8",
    )

    temperature = temperature.merge(
        weights[["city_key", "population", "weight", "city_original"]].rename(
            columns={"city_original": "population_city_original"}
        ),
        on="city_key",
        how="left",
        validate="many_to_one",
    )
    if temperature["population"].isna().any():
        raise ValueError("Her temperature sehrine nufus atanmadi; merge sonrasi null degerler var.")

    province_hourly = (
        temperature.groupby(["source_year", "datetime", "city_key"], as_index=False)
        .agg(
            temperature=("temperature", "mean"),
            population=("population", "first"),
            global_weight=("weight", "first"),
            population_city_original=("population_city_original", "first"),
            station_count=("city_original", "nunique"),
        )
        .sort_values(["source_year", "city_key", "datetime"])
        .reset_index(drop=True)
    )

    observed_counts = (
        province_hourly.groupby(["source_year", "city_key"], as_index=False)
        .agg(observed_hours=("datetime", "count"))
        .sort_values(["source_year", "observed_hours", "city_key"])
        .reset_index(drop=True)
    )
    observed_counts.to_csv(
        output_dir / "population_weighted_observed_hour_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_city_keys = weights["city_key"].tolist()
    total_population = float(weights["population"].sum())
    all_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    yearly_weight_rows: list[pd.DataFrame] = []

    for year in years:
        year_df = province_hourly.loc[province_hourly["source_year"] == year].copy()
        present_keys = set(year_df["city_key"].unique())
        missing_keys = sorted(set(all_city_keys) - present_keys)
        strict_mode_used = not missing_keys

        if require_full_target_coverage and missing_keys:
            pd.DataFrame(coverage_rows).to_csv(
                output_dir / "population_weighted_year_coverage.csv",
                index=False,
                encoding="utf-8-sig",
            )
            raise ValueError(
                f"{year} icin sicaklik dosyasinda hic gozlenmeyen population unit'leri var: {missing_keys}"
            )

        available_weights = (
            weights.loc[weights["city_key"].isin(present_keys), ["city_key", "city_original", "population"]]
            .copy()
            .sort_values("city_key")
            .reset_index(drop=True)
        )
        available_population_sum = float(available_weights["population"].sum())
        available_weights["adjusted_weight"] = available_weights["population"] / available_population_sum
        available_weights["year"] = year
        available_weights["strict_mode_used"] = strict_mode_used
        available_weights["weighting_note"] = (
            "strict_full_coverage"
            if strict_mode_used
            else "renormalized_available_province_weights"
        )
        yearly_weight_rows.append(available_weights.copy())

        year_df = year_df.merge(
            available_weights[["city_key", "adjusted_weight", "strict_mode_used", "weighting_note"]],
            on="city_key",
            how="inner",
            validate="many_to_one",
        )

        hourly_coverage = (
            year_df.groupby("datetime", as_index=False)
            .agg(
                observed_province_count=("city_key", "nunique"),
                hourly_weight_sum=("adjusted_weight", "sum"),
            )
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        hourly_coverage["year"] = year
        hourly_coverage.to_csv(
            output_dir / f"population_weighted_hourly_coverage_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        coverage_rows.append(
            {
                "year": year,
                "expected_city_count": len(all_city_keys),
                "present_city_count": len(present_keys),
                "missing_city_count": len(missing_keys),
                "strict_mode_used": strict_mode_used,
                "weighting_note": (
                    "strict_full_coverage"
                    if strict_mode_used
                    else "renormalized_available_province_weights"
                ),
                "original_population_coverage_ratio": available_population_sum / total_population,
                "adjusted_weight_sum": float(available_weights["adjusted_weight"].sum()),
                "min_hourly_observed_province_count": int(hourly_coverage["observed_province_count"].min()),
                "max_hourly_observed_province_count": int(hourly_coverage["observed_province_count"].max()),
                "min_hourly_weight_sum": float(hourly_coverage["hourly_weight_sum"].min()),
                "max_hourly_weight_sum": float(hourly_coverage["hourly_weight_sum"].max()),
                "missing_city_keys": " | ".join(missing_keys),
            }
        )
        all_frames.append(year_df)

    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "population_weighted_year_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(yearly_weight_rows, ignore_index=True).to_csv(
        output_dir / "population_weighted_yearly_adjusted_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    methodology_lines = [
        "Population-Weighted HDD/CDD Methodology Note",
        "",
        "Base thresholds:",
        f"- HDD base: {BASE_HDD_THRESHOLD}",
        f"- CDD base: {BASE_CDD_THRESHOLD}",
        "",
        "Method:",
        "- Province-level HDD and CDD are computed first from observed province temperatures.",
        "- National weighted HDD/CDD is then aggregated using population-based weights.",
        "- Simple averaging is not used.",
        "- Weighted average temperature is not used as an intermediate step.",
        "",
        "Coverage rule:",
        "- For years with full provincial coverage, strict full-coverage weights are used.",
        "- For years with incomplete provincial temperature coverage, the national weighted HDD/CDD series is computed using population weights renormalized over the provinces available in that year.",
        "- This avoids artificial temperature imputation while keeping the weighting structure population-based.",
        "- No missing province temperatures are imputed.",
    ]
    (output_dir / "population_weighted_methodology_note.txt").write_text(
        "\n".join(methodology_lines),
        encoding="utf-8",
    )

    completed_province_hourly = pd.concat(all_frames, ignore_index=True)
    completed_province_hourly["HDD"] = np.maximum(
        BASE_HDD_THRESHOLD - completed_province_hourly["temperature"],
        0,
    )
    completed_province_hourly["CDD"] = np.maximum(
        completed_province_hourly["temperature"] - BASE_CDD_THRESHOLD,
        0,
    )
    completed_province_hourly["weighted_HDD_component"] = (
        completed_province_hourly["adjusted_weight"] * completed_province_hourly["HDD"]
    )
    completed_province_hourly["weighted_CDD_component"] = (
        completed_province_hourly["adjusted_weight"] * completed_province_hourly["CDD"]
    )

    final_series = (
        completed_province_hourly.groupby("datetime", as_index=False)
        .agg(
            weighted_HDD=("weighted_HDD_component", "sum"),
            weighted_CDD=("weighted_CDD_component", "sum"),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    completed_province_hourly.to_csv(
        output_dir / "population_weighted_province_hourly_hdd_cdd.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_series.to_csv(
        output_dir / "population_weighted_hourly_hdd_cdd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return final_series, {
        "manual_city_mapping": mapping_source,
        "temperature_unique_city_count": int(unique_temperature_cities["city_original"].nunique()),
        "population_unique_city_count": int(unique_population_cities["city_original"].nunique()),
        "matched_city_count": int(matched_cities["city_key"].nunique()),
        "weights_path": str(output_dir / "population_weighted_city_weights.csv"),
        "yearly_adjusted_weights_path": str(output_dir / "population_weighted_yearly_adjusted_weights.csv"),
        "coverage_report_path": str(output_dir / "population_weighted_year_coverage.csv"),
        "methodology_note_path": str(output_dir / "population_weighted_methodology_note.txt"),
        "final_series_path": str(output_dir / "population_weighted_hourly_hdd_cdd.csv"),
    }


def create_month_dummies(df: pd.DataFrame) -> pd.DataFrame:
    columns = [f"month_{month}" for month in range(1, 13) if month != 9]
    dummies = pd.get_dummies(df["month"], prefix="month", dtype=int)
    return dummies.reindex(columns=columns, fill_value=0)


def fit_scaler(frame: pd.DataFrame, columns: list[str]) -> ZScoreScaler:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        means[column] = float(numeric.mean())
        std = float(numeric.std(ddof=0))
        stds[column] = std if not np.isnan(std) and std != 0.0 else 1.0
    return ZScoreScaler(means=means, stds=stds, columns=tuple(columns))


def validate_lasso_dependency() -> None:
    if (
        LassoCV is None
        or RidgeCV is None
        or ElasticNetCV is None
        or r2_score is None
    ):
        raise ImportError(
            "LASSO / Ridge / Elastic Net analizi icin scikit-learn gerekli. "
            "Lutfen `pip install scikit-learn` ya da `pip install -r requirements.txt` calistir."
        ) from SKLEARN_IMPORT_ERROR


def prepare_feature_frame(df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    feature_columns = list(spec["features"])
    features_raw = df[feature_columns].copy()
    if bool(spec["use_month_dummies"]):
        features_raw = pd.concat([features_raw, create_month_dummies(df)], axis=1)
    return features_raw


def prepare_lasso_source_frame(
    df: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_main_raw = prepare_feature_frame(df, spec)
    candidate_columns = list(base_main_raw.columns)
    allow_month_candidates = not bool(spec["use_month_dummies"])

    for left, right in FORCED_INTERACTION_PAIRS:
        if not allow_month_candidates and "month" in {left, right}:
            continue
        if left not in candidate_columns:
            candidate_columns.append(left)
        if right not in candidate_columns:
            candidate_columns.append(right)

    additional_candidate_columns = ["hour"]
    if allow_month_candidates:
        additional_candidate_columns.append("month")

    for column in additional_candidate_columns:
        if column not in candidate_columns:
            candidate_columns.append(column)

    source_columns = [
        column
        for column in candidate_columns
        if column in df.columns and not column.startswith("month_")
    ]
    lasso_source_raw = df[source_columns].copy()
    return base_main_raw, lasso_source_raw


def center_and_scale_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    scaler = fit_scaler(numeric_frame, list(numeric_frame.columns))
    scaled = scaler.transform(numeric_frame).astype(float)
    scaler_table = pd.DataFrame(
        {
            "variable": list(numeric_frame.columns),
            "mean": [scaler.means[column] for column in numeric_frame.columns],
            "std": [scaler.stds[column] for column in numeric_frame.columns],
        }
    )
    return scaled, scaler_table


def build_polynomial_candidates(feature_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    polynomial_columns: dict[str, pd.Series] = {}

    for column in numeric_frame.columns:
        if column not in POLYNOMIAL_SOURCE_COLUMNS:
            continue

        polynomial_name = f"{column}_sq"
        values = numeric_frame[column] ** 2
        non_null = values.dropna()
        if non_null.empty:
            continue

        std = float(non_null.std(ddof=0))
        if np.isnan(std) or std == 0.0:
            continue

        polynomial_columns[polynomial_name] = values

    if not polynomial_columns:
        return pd.DataFrame(index=feature_frame.index)

    return pd.DataFrame(polynomial_columns, index=feature_frame.index)


def build_interaction_candidates(feature_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    interaction_columns: dict[str, pd.Series] = {}

    for left, right in combinations(numeric_frame.columns, 2):
        if frozenset({left, right}) in STRUCTURALLY_INVALID_INTERACTION_PAIRS:
            continue
        interaction_name = f"{left}_x_{right}"
        interaction_values = numeric_frame[left] * numeric_frame[right]
        non_null = interaction_values.dropna()
        if non_null.empty:
            continue

        std = float(non_null.std(ddof=0))
        if np.isnan(std) or std == 0.0:
            continue

        interaction_columns[interaction_name] = interaction_values

    if not interaction_columns:
        return pd.DataFrame(index=feature_frame.index)

    return pd.DataFrame(interaction_columns, index=feature_frame.index)


def prune_design_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    kept_columns: list[str] = []
    dropped_rows: list[dict[str, object]] = []

    for column in numeric_frame.columns:
        series = numeric_frame[column]
        non_null = series.dropna()
        if non_null.empty:
            dropped_rows.append({"variable": column, "drop_reason": "all_null"})
            continue

        std = float(non_null.std(ddof=0))
        if np.isnan(std) or std <= DESIGN_STD_TOLERANCE:
            dropped_rows.append(
                {"variable": column, "drop_reason": "low_variance_or_constant"}
            )
            continue

        candidate_columns = kept_columns + [column]
        candidate_frame = numeric_frame[candidate_columns].dropna()
        if kept_columns and not candidate_frame.empty:
            rank_input = np.column_stack(
                [np.ones(len(candidate_frame), dtype=float), candidate_frame.values]
            )
            rank = np.linalg.matrix_rank(
                rank_input,
                tol=DESIGN_RANK_TOLERANCE,
            )
            if rank < len(candidate_columns) + 1:
                dropped_rows.append(
                    {
                        "variable": column,
                        "drop_reason": "exact_linear_dependency_with_intercept",
                    }
                )
                continue

        kept_columns.append(column)

    pruned = numeric_frame[kept_columns].copy()
    dropped = pd.DataFrame(dropped_rows, columns=["variable", "drop_reason"])
    return pruned, dropped


def get_hierarchy_protected_columns(columns: list[str]) -> set[str]:
    current_columns = set(columns)
    protected: set[str] = set()

    for column in columns:
        term_type, component_1, component_2 = parse_candidate_term(column)
        if term_type == "polynomial":
            if component_1 in current_columns:
                protected.add(component_1)
        elif term_type == "interaction":
            if component_1 in current_columns:
                protected.add(component_1)
            if component_2 in current_columns:
                protected.add(component_2)

    return protected


def prune_by_p_value(
    df: pd.DataFrame,
    y_col: str,
    design_matrix: pd.DataFrame,
    threshold: float = P_VALUE_SELECTION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_design = design_matrix.copy()
    dropped_rows: list[dict[str, object]] = []

    while current_design.shape[1] > 0:
        y = df[y_col].astype(float).rename(y_col)
        X = sm.add_constant(current_design.astype(float), has_constant="add")
        model_data = pd.concat([y, X], axis=1).dropna()
        y_clean = model_data[y_col]
        X_clean = model_data.drop(columns=[y_col])
        model = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")

        p_values = model.pvalues.drop(labels=["const"], errors="ignore")
        protected_columns = get_hierarchy_protected_columns(list(current_design.columns))
        removable = p_values[
            (p_values > threshold) & (~p_values.index.isin(protected_columns))
        ]
        if removable.empty:
            break

        drop_variable = str(removable.sort_values(ascending=False).index[0])
        dropped_rows.append(
            {
                "variable": drop_variable,
                "drop_reason": "p_value_above_threshold",
                "p_value": float(removable.loc[drop_variable]),
            }
        )
        current_design = current_design.drop(columns=[drop_variable])

    dropped = pd.DataFrame(
        dropped_rows,
        columns=["variable", "drop_reason", "p_value"],
    )
    return current_design, dropped


def build_identity_scaler_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": columns,
            "mean": np.zeros(len(columns), dtype=float),
            "std": np.ones(len(columns), dtype=float),
        }
    )


def build_vif_table(
    vif_input: pd.DataFrame,
    model_id: str,
    prompt: str,
) -> pd.DataFrame:
    if vif_input.shape[1] == 0:
        return pd.DataFrame(columns=["model_id", "prompt", "variable", "vif"])

    return pd.DataFrame(
        {
            "model_id": model_id,
            "prompt": prompt,
            "variable": vif_input.columns,
            "vif": [
                variance_inflation_factor(vif_input.values, i)
                for i in range(vif_input.shape[1])
            ],
        }
    ).sort_values("vif", ascending=False)


def fit_ols_on_design_matrix(
    df: pd.DataFrame,
    y_col: str,
    design_matrix: pd.DataFrame,
    model_id: str,
    prompt: str,
    scaler_table: pd.DataFrame | None = None,
) -> tuple[
    RegressionResultsWrapper,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    y = df[y_col].astype(float).rename(y_col)
    X = sm.add_constant(design_matrix.astype(float), has_constant="add")
    model_data = pd.concat([y, X], axis=1).dropna()
    y_clean = model_data[y_col]
    X_clean = model_data.drop(columns=[y_col])
    model = sm.OLS(y_clean, X_clean).fit(cov_type="HC1")

    fitted = df.loc[model_data.index].copy()
    fitted[f"fitted_{y_col}"] = model.predict(X_clean)

    vif_table = build_vif_table(
        X_clean.drop(columns=["const"], errors="ignore"),
        model_id=model_id,
        prompt=prompt,
    )
    ols_table = pd.DataFrame(
        {
            "model_id": model_id,
            "prompt": prompt,
            "variable": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )
    diagnostics = pd.DataFrame(
        [
            {"model_id": model_id, "prompt": prompt, "metric": "dependent_variable", "value": y_col},
            {"model_id": model_id, "prompt": prompt, "metric": "plot_year", "value": int(df["year"].iloc[0])},
            {"model_id": model_id, "prompt": prompt, "metric": "n_obs", "value": len(model_data)},
            {"model_id": model_id, "prompt": prompt, "metric": "durbin_watson", "value": durbin_watson(model.resid)},
            {"model_id": model_id, "prompt": prompt, "metric": "condition_number", "value": np.linalg.cond(X_clean)},
            {"model_id": model_id, "prompt": prompt, "metric": "r_squared", "value": model.rsquared},
            {"model_id": model_id, "prompt": prompt, "metric": "adj_r_squared", "value": model.rsquared_adj},
        ]
    )

    if scaler_table is None:
        scaler_table = pd.DataFrame(columns=["variable", "mean", "std"])

    scaler_table_out = scaler_table.copy()
    scaler_table_out.insert(0, "prompt", prompt)
    scaler_table_out.insert(0, "model_id", model_id)
    return model, fitted, ols_table, diagnostics, scaler_table_out, vif_table


def fit_penalized_model(
    df: pd.DataFrame,
    y_col: str,
    design_matrix: pd.DataFrame,
    model_id: str,
    prompt: str,
    model_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[y_col].astype(float).rename(y_col)
    model_data = pd.concat([y, design_matrix.astype(float)], axis=1).dropna()
    y_clean = model_data[y_col]
    X_clean = model_data.drop(columns=[y_col])

    if model_kind == "ridge":
        penalized_model = RidgeCV(alphas=RIDGE_ALPHA_GRID, cv=5)
        penalized_model.fit(X_clean, y_clean)
        best_alpha = float(penalized_model.alpha_)
        best_l1_ratio = np.nan
    elif model_kind == "elastic_net":
        penalized_model = ElasticNetCV(
            l1_ratio=ELASTIC_NET_L1_RATIOS,
            cv=5,
            random_state=LASSO_RANDOM_STATE,
            max_iter=LASSO_MAX_ITER,
        )
        penalized_model.fit(X_clean, y_clean)
        best_alpha = float(penalized_model.alpha_)
        best_l1_ratio = float(penalized_model.l1_ratio_)
    else:
        raise ValueError(f"Desteklenmeyen penalized model tipi: {model_kind}")

    predictions = penalized_model.predict(X_clean)
    coef_table = pd.DataFrame(
        {
            "model_id": model_id,
            "prompt": prompt,
            "model_kind": model_kind,
            "variable": X_clean.columns,
            "coefficient": penalized_model.coef_,
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

    diagnostics = pd.DataFrame(
        [
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "dependent_variable", "value": y_col},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "plot_year", "value": int(df["year"].iloc[0])},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "n_obs", "value": len(model_data)},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "condition_number", "value": np.linalg.cond(X_clean)},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "r_squared", "value": r2_score(y_clean, predictions)},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "alpha", "value": best_alpha},
            {"model_id": model_id, "prompt": prompt, "model_kind": model_kind, "metric": "l1_ratio", "value": best_l1_ratio},
        ]
    )
    return coef_table, diagnostics


def build_model_specs() -> list[dict[str, object]]:
    return [
        {
            "model_id": "model_1",
            "prompt": "TABLE 1 - consumption ~ HDD + CDD",
            "y_col": "consumption",
            "features": ["HDD", "CDD"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_2",
            "prompt": "TABLE 2 - consumption ~ HDD + CDD + night dummy",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_3",
            "prompt": "TABLE 3 - consumption ~ HDD + CDD + night dummy + weekend dummy",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_4",
            "prompt": "TABLE 4 - consumption ~ HDD + CDD + night dummy + weekend dummy + month dummies",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend"],
            "use_month_dummies": True,
        },
        {
            "model_id": "model_5",
            "prompt": "TABLE 5 - consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_6",
            "prompt": "TABLE 6 - consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR + CUR",
            "y_col": "consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR", "CUR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_7",
            "prompt": "TABLE 7 - log_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "log_consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
        {
            "model_id": "model_8",
            "prompt": "TABLE 8 - sqrt_consumption ~ HDD + CDD + night dummy + weekend dummy + PMI + IR",
            "y_col": "sqrt_consumption",
            "features": ["HDD", "CDD", "night", "weekend", "PMI", "IR"],
            "use_month_dummies": False,
        },
    ]


def fit_ols_from_feature_frame(
    df: pd.DataFrame,
    y_col: str,
    features_raw: pd.DataFrame,
    model_id: str,
    prompt: str,
) -> tuple[
    RegressionResultsWrapper,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    continuous_columns = [
        column for column in features_raw.columns if column in ALL_CONTINUOUS_COLUMNS
    ]
    interaction_columns = [
        column for column in features_raw.columns if "_x_" in column
    ]
    columns_to_scale = continuous_columns + interaction_columns
    scaler = fit_scaler(features_raw, columns_to_scale)
    features_scaled = scaler.transform(features_raw)

    X_raw = sm.add_constant(features_raw.astype(float), has_constant="add")
    X_scaled = sm.add_constant(features_scaled.astype(float), has_constant="add")
    y = df[y_col].astype(float).rename(y_col)

    model_data = pd.concat(
        [y, X_raw.add_prefix("raw__"), X_scaled.add_prefix("scaled__")],
        axis=1,
    ).dropna()

    X_raw_clean = model_data[[f"raw__{col}" for col in X_raw.columns]].rename(
        columns={f"raw__{col}": col for col in X_raw.columns}
    )
    X_scaled_clean = model_data[[f"scaled__{col}" for col in X_scaled.columns]].rename(
        columns={f"scaled__{col}": col for col in X_scaled.columns}
    )
    y_clean = model_data[y_col]

    model = sm.OLS(y_clean, X_scaled_clean).fit(cov_type="HC1")

    fitted = df.loc[model_data.index].copy()
    fitted[f"fitted_{y_col}"] = model.predict(X_scaled_clean)

    vif_input = X_scaled_clean.drop(columns=["const"], errors="ignore")
    if vif_input.shape[1] > 0:
        vif_table = pd.DataFrame(
            {
                "model_id": model_id,
                "prompt": prompt,
                "variable": vif_input.columns,
                "vif": [
                    variance_inflation_factor(vif_input.values, i)
                    for i in range(vif_input.shape[1])
                ],
            }
        ).sort_values("vif", ascending=False)
    else:
        vif_table = pd.DataFrame(
            columns=["model_id", "prompt", "variable", "vif"]
        )

    ols_table = pd.DataFrame(
        {
            "model_id": model_id,
            "prompt": prompt,
            "variable": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )

    diagnostics = pd.DataFrame(
        [
            {"model_id": model_id, "prompt": prompt, "metric": "dependent_variable", "value": y_col},
            {"model_id": model_id, "prompt": prompt, "metric": "plot_year", "value": int(df["year"].iloc[0])},
            {"model_id": model_id, "prompt": prompt, "metric": "n_obs", "value": len(model_data)},
            {"model_id": model_id, "prompt": prompt, "metric": "durbin_watson", "value": durbin_watson(model.resid)},
            {"model_id": model_id, "prompt": prompt, "metric": "raw_condition_number", "value": np.linalg.cond(X_raw_clean)},
            {"model_id": model_id, "prompt": prompt, "metric": "scaled_condition_number", "value": np.linalg.cond(X_scaled_clean)},
            {"model_id": model_id, "prompt": prompt, "metric": "r_squared", "value": model.rsquared},
            {"model_id": model_id, "prompt": prompt, "metric": "adj_r_squared", "value": model.rsquared_adj},
            {"model_id": model_id, "prompt": prompt, "metric": "interaction_term_count", "value": len(interaction_columns)},
        ]
    )

    scaler_table = pd.DataFrame(
        {
            "model_id": model_id,
            "prompt": prompt,
            "variable": columns_to_scale,
            "mean": [scaler.means[column] for column in columns_to_scale],
            "std": [scaler.stds[column] for column in columns_to_scale],
        }
    )

    return model, fitted, ols_table, diagnostics, scaler_table, vif_table


def fit_single_model(
    df: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[
    RegressionResultsWrapper,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    features_raw = prepare_feature_frame(df, spec)
    model, fitted, ols_table, diagnostics, scaler_table, vif_table = fit_ols_from_feature_frame(
        df=df,
        y_col=str(spec["y_col"]),
        features_raw=features_raw,
        model_id=str(spec["model_id"]),
        prompt=str(spec["prompt"]),
    )
    diagnostics = pd.concat(
        [
            diagnostics,
            pd.DataFrame(
                [
                    {
                        "model_id": str(spec["model_id"]),
                        "prompt": str(spec["prompt"]),
                        "metric": "month_dummy_in_model",
                        "value": int(bool(spec["use_month_dummies"])),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return model, fitted, ols_table, diagnostics, scaler_table, vif_table


def parse_candidate_term(variable: str) -> tuple[str, str, str]:
    if variable.endswith("_sq"):
        return "polynomial", variable[:-3], ""
    if "_x_" in variable:
        left, right = variable.split("_x_", 1)
        return "interaction", left, right
    return "other", variable, ""


def is_month_dummy_collinear_term(
    variable: str,
    spec: dict[str, object],
) -> bool:
    if not bool(spec["use_month_dummies"]):
        return False

    term_type, component_1, component_2 = parse_candidate_term(variable)
    if term_type == "polynomial":
        return component_1 in MONTH_LEVEL_COLUMNS
    if term_type == "interaction":
        return (
            component_1 in MONTH_LEVEL_COLUMNS
            and component_2 in MONTH_LEVEL_COLUMNS
        )
    return False


def run_lasso_interaction_selection(
    df: pd.DataFrame,
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_lasso_dependency()

    model_id = str(spec["model_id"])
    prompt = str(spec["prompt"])
    y_col = str(spec["y_col"])
    _, lasso_source_raw = prepare_lasso_source_frame(df, spec)
    interaction_candidates = build_interaction_candidates(lasso_source_raw)
    polynomial_candidates = build_polynomial_candidates(lasso_source_raw)
    candidate_terms = pd.concat(
        [interaction_candidates, polynomial_candidates],
        axis=1,
    )
    candidate_terms = candidate_terms[
        [
            column
            for column in candidate_terms.columns
            if not is_month_dummy_collinear_term(column, spec)
        ]
    ]

    if candidate_terms.empty:
        empty_candidates = pd.DataFrame(
            columns=[
                "model_id",
                "prompt",
                "dependent_variable",
                "variable",
                "term_type",
                "component_1",
                "component_2",
                "lasso_coefficient",
                "abs_lasso_coefficient",
                "selected",
                "forced_priority",
            ]
        )
        metadata = pd.DataFrame(
            [
                {"model_id": model_id, "prompt": prompt, "metric": "dependent_variable", "value": y_col},
                {"model_id": model_id, "prompt": prompt, "metric": "candidate_interaction_count", "value": 0},
                {"model_id": model_id, "prompt": prompt, "metric": "candidate_polynomial_count", "value": 0},
                {"model_id": model_id, "prompt": prompt, "metric": "selected_interaction_count", "value": 0},
                {"model_id": model_id, "prompt": prompt, "metric": "selected_polynomial_count", "value": 0},
            ]
        )
        return empty_candidates, empty_candidates.copy(), metadata

    y = df[y_col].astype(float).rename(y_col)
    lasso_feature_frame = pd.concat([lasso_source_raw, candidate_terms], axis=1)
    lasso_features, _ = center_and_scale_frame(lasso_feature_frame)
    model_data = pd.concat([y, lasso_features], axis=1).dropna()
    X_clean = model_data.drop(columns=[y_col]).astype(float)
    y_clean = model_data[y_col]

    lasso_model = LassoCV(
        cv=5,
        random_state=LASSO_RANDOM_STATE,
        max_iter=LASSO_MAX_ITER,
    )
    lasso_model.fit(X_clean, y_clean)

    all_coefficients = pd.DataFrame(
        {
            "variable": X_clean.columns,
            "lasso_coefficient": lasso_model.coef_,
        }
    )
    all_coefficients["abs_lasso_coefficient"] = all_coefficients["lasso_coefficient"].abs()
    all_coefficients["selected"] = (
        all_coefficients["abs_lasso_coefficient"] > LASSO_SELECTION_TOLERANCE
    ).astype(int)
    parsed = all_coefficients["variable"].map(parse_candidate_term)
    all_coefficients["term_type"] = parsed.map(lambda item: item[0])
    all_coefficients["component_1"] = parsed.map(lambda item: item[1])
    all_coefficients["component_2"] = parsed.map(lambda item: item[2])
    all_coefficients["forced_priority"] = all_coefficients["variable"].isin(
        [f"{left}_x_{right}" for left, right in FORCED_INTERACTION_PAIRS]
        + [f"{right}_x_{left}" for left, right in FORCED_INTERACTION_PAIRS]
    ).astype(int)

    candidate_table = (
        all_coefficients[
            all_coefficients["term_type"].isin(["interaction", "polynomial"])
        ]
        .copy()
        .sort_values(
            ["selected", "forced_priority", "abs_lasso_coefficient"],
            ascending=[False, False, False],
        )
    )
    candidate_table.insert(0, "dependent_variable", y_col)
    candidate_table.insert(0, "prompt", prompt)
    candidate_table.insert(0, "model_id", model_id)

    selected_interactions = (
        candidate_table[candidate_table["selected"] == 1]
        .copy()
        .reset_index(drop=True)
    )

    metadata = pd.DataFrame(
        [
            {"model_id": model_id, "prompt": prompt, "metric": "dependent_variable", "value": y_col},
            {"model_id": model_id, "prompt": prompt, "metric": "candidate_interaction_count", "value": int((candidate_table["term_type"] == "interaction").sum())},
            {"model_id": model_id, "prompt": prompt, "metric": "candidate_polynomial_count", "value": int((candidate_table["term_type"] == "polynomial").sum())},
            {"model_id": model_id, "prompt": prompt, "metric": "selected_interaction_count", "value": len(selected_interactions)},
            {"model_id": model_id, "prompt": prompt, "metric": "selected_polynomial_count", "value": int((selected_interactions["term_type"] == "polynomial").sum())},
            {"model_id": model_id, "prompt": prompt, "metric": "lasso_alpha", "value": lasso_model.alpha_},
            {"model_id": model_id, "prompt": prompt, "metric": "lasso_intercept", "value": lasso_model.intercept_},
            {"model_id": model_id, "prompt": prompt, "metric": "lasso_r_squared", "value": r2_score(y_clean, lasso_model.predict(X_clean))},
        ]
    )
    return candidate_table, selected_interactions, metadata


def fit_lasso_augmented_model(
    df: pd.DataFrame,
    spec: dict[str, object],
    selected_interactions: pd.DataFrame,
) -> dict[str, object] | None:
    if selected_interactions.empty:
        return None

    base_main_raw, _ = prepare_lasso_source_frame(df, spec)
    required_main_columns = list(base_main_raw.columns)
    extra_main_columns: list[str] = []
    use_month_dummies = bool(spec["use_month_dummies"])

    for _, row in selected_interactions.iterrows():
        term_type = str(row["term_type"])
        component_1 = str(row["component_1"])
        component_2 = str(row["component_2"])
        if term_type == "polynomial":
            if (
                component_1 not in required_main_columns
                and component_1 in df.columns
                and not (
                    use_month_dummies and component_1 in MONTH_LEVEL_COLUMNS
                )
            ):
                extra_main_columns.append(component_1)
        elif term_type == "interaction":
            for component in (component_1, component_2):
                if (
                    component not in required_main_columns
                    and component in df.columns
                    and not (
                        use_month_dummies and component in MONTH_LEVEL_COLUMNS
                    )
                ):
                    extra_main_columns.append(component)

    for column in extra_main_columns:
        if column not in required_main_columns:
            required_main_columns.append(column)

    final_design_raw = pd.DataFrame(index=df.index)
    for column in required_main_columns:
        if column in base_main_raw.columns:
            final_design_raw[column] = pd.to_numeric(
                base_main_raw[column], errors="coerce"
            )
        else:
            final_design_raw[column] = pd.to_numeric(df[column], errors="coerce")

    for _, row in selected_interactions.iterrows():
        variable = str(row["variable"])
        term_type = str(row["term_type"])
        component_1 = str(row["component_1"])
        component_2 = str(row["component_2"])
        if term_type == "polynomial":
            if component_1 in final_design_raw.columns:
                final_design_raw[variable] = final_design_raw[component_1] ** 2
        elif term_type == "interaction":
            if (
                component_1 in final_design_raw.columns
                and component_2 in final_design_raw.columns
            ):
                final_design_raw[variable] = (
                    final_design_raw[component_1] * final_design_raw[component_2]
                )

    final_design_raw, dropped_design_terms = prune_design_columns(final_design_raw)
    final_design, scaler_table = center_and_scale_frame(final_design_raw)
    final_design, dropped_pvalue_terms = prune_by_p_value(
        df=df,
        y_col=str(spec["y_col"]),
        design_matrix=final_design,
    )
    scaler_table = scaler_table[
        scaler_table["variable"].isin(final_design.columns)
    ].reset_index(drop=True)
    final_design_raw = final_design_raw[
        [column for column in final_design.columns if column in final_design_raw.columns]
    ].copy()

    if final_design.empty:
        return None

    if final_design_raw.empty:
        return None

    model_id = f"{spec['model_id']}_lasso_final"
    prompt = (
        f"{spec['prompt']} | selection standardized + "
        "LASSO-selected terms + p-value filtered + raw-scale final OLS"
    )
    ols_outputs = fit_ols_on_design_matrix(
        df=df,
        y_col=str(spec["y_col"]),
        design_matrix=final_design_raw,
        model_id=str(model_id),
        prompt=str(prompt),
        scaler_table=scaler_table,
    )
    (
        ols_model,
        fitted_df,
        ols_table,
        diagnostics,
        scaler_table_out,
        vif_table,
    ) = ols_outputs
    diagnostics = pd.concat(
        [
            diagnostics,
            pd.DataFrame(
                [
                    {
                        "model_id": str(model_id),
                        "prompt": str(prompt),
                        "metric": "dropped_design_term_count",
                        "value": int(len(dropped_design_terms)),
                    },
                    {
                        "model_id": str(model_id),
                        "prompt": str(prompt),
                        "metric": "dropped_pvalue_term_count",
                        "value": int(len(dropped_pvalue_terms)),
                    },
                    {
                        "model_id": str(model_id),
                        "prompt": str(prompt),
                        "metric": "selection_scaled_condition_number",
                        "value": float(
                            np.linalg.cond(
                                sm.add_constant(
                                    final_design.astype(float),
                                    has_constant="add",
                                )
                            )
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ols_outputs = (
        ols_model,
        fitted_df,
        ols_table,
        diagnostics,
        scaler_table_out,
        vif_table,
    )
    marginal_effect_frame = final_design_raw.loc[fitted_df.index].copy()
    marginal_effect_scaler_table = build_identity_scaler_table(
        list(marginal_effect_frame.columns)
    )

    def build_marginal_effect_outputs(base_variable: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        component_frame = compute_marginal_effect_component_frame(
            raw_design_frame=marginal_effect_frame,
            model=ols_model,
            scaler_table=marginal_effect_scaler_table,
            base_variable=base_variable,
        )
        marginal_effect_series = component_frame["total_marginal_effect"]
        summary = summarize_monthly_marginal_effects(
            fitted_df=fitted_df,
            marginal_effect=marginal_effect_series,
            model_id=str(model_id),
            prompt=str(prompt),
            y_col=str(spec["y_col"]),
            base_variable=base_variable,
        )
        component_summary = summarize_monthly_marginal_effect_components(
            fitted_df=fitted_df,
            component_frame=component_frame,
            model_id=str(model_id),
            prompt=str(prompt),
            y_col=str(spec["y_col"]),
            base_variable=base_variable,
        )
        return summary, component_summary

    cdd_marginal_effect_summary, cdd_marginal_effect_component_summary = (
        build_marginal_effect_outputs("CDD")
    )
    hdd_marginal_effect_summary, hdd_marginal_effect_component_summary = (
        build_marginal_effect_outputs("HDD")
    )
    ridge_coef_table, ridge_diagnostics = fit_penalized_model(
        df=df,
        y_col=str(spec["y_col"]),
        design_matrix=final_design,
        model_id=str(model_id),
        prompt=str(prompt),
        model_kind="ridge",
    )
    elastic_net_coef_table, elastic_net_diagnostics = fit_penalized_model(
        df=df,
        y_col=str(spec["y_col"]),
        design_matrix=final_design,
        model_id=str(model_id),
        prompt=str(prompt),
        model_kind="elastic_net",
    )
    return {
        "ols": ols_outputs,
        "ridge_coef_table": ridge_coef_table,
        "ridge_diagnostics": ridge_diagnostics,
        "elastic_net_coef_table": elastic_net_coef_table,
        "elastic_net_diagnostics": elastic_net_diagnostics,
        "cdd_marginal_effect_summary": cdd_marginal_effect_summary,
        "cdd_marginal_effect_component_summary": cdd_marginal_effect_component_summary,
        "hdd_marginal_effect_summary": hdd_marginal_effect_summary,
        "hdd_marginal_effect_component_summary": hdd_marginal_effect_component_summary,
        "design_columns": pd.concat(
            [
                pd.DataFrame({"variable": final_design.columns, "status": "kept"}),
                dropped_design_terms.assign(status="dropped_design"),
                dropped_pvalue_terms.assign(status="dropped_pvalue"),
            ],
            ignore_index=True,
        ),
    }


def save_year_plots(year_df: pd.DataFrame, year: int, fitted_column: str) -> list[Path]:
    plot_paths: list[Path] = []

    if year_df.empty:
        return plot_paths

    daily = (
        year_df.set_index("datetime")[["consumption", fitted_column]]
        .resample("D")
        .mean()
    )
    ax = daily.plot(figsize=(14, 5), linewidth=1.2)
    ax.set_title(f"{year} Gunluk Ortalama Tuketim: Gercek vs OLS")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_daily = FIGURES_DIR / f"{year}_gunluk_gercek_vs_ols.png"
    plt.savefig(path_daily, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_daily)

    hourly = year_df.groupby("hour", as_index=True)["consumption"].mean()
    ax = hourly.plot(figsize=(10, 4), marker="o", color="darkgreen")
    ax.set_title(f"{year} Saatlik Ortalama Tuketim Profili")
    ax.set_xlabel("Saat")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_hourly = FIGURES_DIR / f"{year}_saatlik_tuketim_profili.png"
    plt.savefig(path_hourly, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_hourly)

    monthly = year_df.groupby("month", as_index=True)["consumption"].mean()
    ax = monthly.plot(kind="bar", figsize=(10, 4), color="steelblue")
    ax.set_title(f"{year} Aylik Ortalama Tuketim")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Tuketim (MWh)")
    plt.tight_layout()
    path_monthly = FIGURES_DIR / f"{year}_aylik_ortalama_tuketim.png"
    plt.savefig(path_monthly, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(path_monthly)

    return plot_paths


def write_vif_txt(path: Path, title: str, vif_table: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        for _, row in vif_table.iterrows():
            handle.write(f"{row['variable']:<20} {row['vif']:.4f}\n")


def write_grouped_vif_txt(path: Path, title: str, vif_table: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        for prompt, group in vif_table.groupby("prompt", sort=False):
            handle.write(f"{prompt}\n")
            handle.write("-" * len(prompt) + "\n")
            for _, row in group.iterrows():
                handle.write(f"{row['variable']:<20} {row['vif']:.4f}\n")
            handle.write("\n")


def write_selected_interactions_txt(
    path: Path,
    title: str,
    interaction_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if interaction_table.empty:
            handle.write("Secilen interaction/polynomial term yok.\n")
            return

        for _, row in interaction_table.iterrows():
            handle.write(
                f"{row['variable']:<35} {str(row['term_type']):<14} {float(row['lasso_coefficient']):>12.6f}\n"
            )


def write_grouped_selected_interactions_txt(
    path: Path,
    title: str,
    interaction_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if interaction_table.empty:
            handle.write("Hicbir modelde secilen interaction/polynomial term yok.\n")
            return

        for prompt, group in interaction_table.groupby("prompt", sort=False):
            handle.write(f"{prompt}\n")
            handle.write("-" * len(prompt) + "\n")
            for _, row in group.iterrows():
                handle.write(
                    f"{row['variable']:<35} {str(row['term_type']):<14} {float(row['lasso_coefficient']):>12.6f}\n"
                )
            handle.write("\n")


def write_lasso_candidate_terms_txt(
    path: Path,
    title: str,
    candidate_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if candidate_table.empty:
            handle.write("Aday interaction/polynomial term yok.\n")
            return

        for _, row in candidate_table.iterrows():
            selected_flag = "SECILDI" if int(row["selected"]) == 1 else "-"
            forced_flag = "ONCELIKLI" if int(row.get("forced_priority", 0)) == 1 else "-"
            handle.write(
                f"{row['variable']:<35} "
                f"{str(row['term_type']):<12} "
                f"{selected_flag:<8} "
                f"{forced_flag:<10} "
                f"{float(row['lasso_coefficient']):>12.6f}\n"
            )


def write_ols_table_txt(
    path: Path,
    title: str,
    ols_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if ols_table.empty:
            handle.write("OLS tablosu bos.\n")
            return

        for _, row in ols_table.iterrows():
            handle.write(
                f"{row['variable']:<30} "
                f"coef={float(row['coefficient']):>12.6f} "
                f"p={float(row['p_value']):>10.6f} "
                f"t={float(row['t_stat']):>10.4f}\n"
            )


def write_diagnostics_txt(
    path: Path,
    title: str,
    diagnostics: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if diagnostics.empty:
            handle.write("Diyagnostik tablo bos.\n")
            return

        for _, row in diagnostics.iterrows():
            value = row["value"]
            if isinstance(value, (float, np.floating)):
                rendered_value = f"{float(value):.6f}"
            else:
                rendered_value = str(value)
            handle.write(f"{row['metric']:<30} {rendered_value}\n")


def write_penalized_coefficients_txt(
    path: Path,
    title: str,
    coefficient_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if coefficient_table.empty:
            handle.write("Katsayi tablosu bos.\n")
            return

        for _, row in coefficient_table.iterrows():
            handle.write(
                f"{row['variable']:<35} {float(row['coefficient']):>12.6f}\n"
            )


def compute_marginal_effect_series(
    raw_design_frame: pd.DataFrame,
    model: RegressionResultsWrapper,
    scaler_table: pd.DataFrame,
    base_variable: str,
) -> pd.Series:
    component_frame = compute_marginal_effect_component_frame(
        raw_design_frame=raw_design_frame,
        model=model,
        scaler_table=scaler_table,
        base_variable=base_variable,
    )
    return component_frame["total_marginal_effect"]


def compute_marginal_effect_component_frame(
    raw_design_frame: pd.DataFrame,
    model: RegressionResultsWrapper,
    scaler_table: pd.DataFrame,
    base_variable: str,
) -> pd.DataFrame:
    std_map = (
        scaler_table[["variable", "std"]]
        .drop_duplicates(subset=["variable"])
        .set_index("variable")["std"]
        .to_dict()
    )
    component_data: dict[str, pd.Series] = {}

    for variable, coefficient in model.params.items():
        if variable == "const":
            continue

        std = float(std_map.get(variable, np.nan))
        if np.isnan(std) or std == 0.0:
            continue

        if variable == base_variable and base_variable in raw_design_frame.columns:
            component_data[variable] = pd.Series(
                float(coefficient) / std,
                index=raw_design_frame.index,
                dtype=float,
            )
            continue

        term_type, component_1, component_2 = parse_candidate_term(str(variable))
        if term_type == "polynomial" and component_1 == base_variable:
            if base_variable in raw_design_frame.columns:
                component_data[variable] = (
                    float(coefficient) / std * 2.0 * raw_design_frame[base_variable]
                )
        elif term_type == "interaction":
            if (
                component_1 == base_variable
                and component_2 in raw_design_frame.columns
            ):
                component_data[variable] = (
                    float(coefficient) / std * raw_design_frame[component_2]
                )
            elif (
                component_2 == base_variable
                and component_1 in raw_design_frame.columns
            ):
                component_data[variable] = (
                    float(coefficient) / std * raw_design_frame[component_1]
                )

    if not component_data:
        return pd.DataFrame(
            {"total_marginal_effect": pd.Series(0.0, index=raw_design_frame.index)}
        )

    component_frame = pd.DataFrame(component_data, index=raw_design_frame.index)
    component_frame["total_marginal_effect"] = component_frame.sum(axis=1)
    return component_frame


def summarize_monthly_marginal_effects(
    fitted_df: pd.DataFrame,
    marginal_effect: pd.Series,
    model_id: str,
    prompt: str,
    y_col: str,
    base_variable: str,
) -> pd.DataFrame:
    working = fitted_df.copy()
    working["marginal_effect"] = pd.to_numeric(marginal_effect, errors="coerce")
    working[base_variable] = pd.to_numeric(working[base_variable], errors="coerce")
    working = working.dropna(subset=["month", "marginal_effect", base_variable])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "prompt",
                "dependent_variable",
                "base_variable",
                "month",
                "month_label",
                "n_obs",
                "avg_marginal_effect",
                "median_marginal_effect",
                "p10_marginal_effect",
                "p90_marginal_effect",
                "min_marginal_effect",
                "max_marginal_effect",
                "positive_share",
                "avg_base_variable",
            ]
        )

    def _quantile_func(q: float):
        return lambda series: float(series.quantile(q))

    summary = (
        working.groupby("month", as_index=False)
        .agg(
            n_obs=("marginal_effect", "size"),
            avg_marginal_effect=("marginal_effect", "mean"),
            median_marginal_effect=("marginal_effect", "median"),
            p10_marginal_effect=("marginal_effect", _quantile_func(0.10)),
            p90_marginal_effect=("marginal_effect", _quantile_func(0.90)),
            min_marginal_effect=("marginal_effect", "min"),
            max_marginal_effect=("marginal_effect", "max"),
            positive_share=("marginal_effect", lambda s: float((s > 0).mean())),
            avg_base_variable=(base_variable, "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    summary["month"] = summary["month"].astype(int)
    summary["month_label"] = summary["month"].astype(str).str.zfill(2)
    summary.insert(0, "base_variable", base_variable)
    summary.insert(0, "dependent_variable", y_col)
    summary.insert(0, "prompt", prompt)
    summary.insert(0, "model_id", model_id)
    return summary


def summarize_monthly_marginal_effect_components(
    fitted_df: pd.DataFrame,
    component_frame: pd.DataFrame,
    model_id: str,
    prompt: str,
    y_col: str,
    base_variable: str,
) -> pd.DataFrame:
    if component_frame.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "prompt",
                "dependent_variable",
                "base_variable",
                "month",
                "month_label",
                "component_variable",
                "avg_component_effect",
            ]
        )

    working = fitted_df[["month"]].copy()
    component_cols = list(component_frame.columns)
    working = pd.concat([working, component_frame[component_cols]], axis=1)
    working = working.dropna(subset=["month"])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "prompt",
                "dependent_variable",
                "base_variable",
                "month",
                "month_label",
                "component_variable",
                "avg_component_effect",
            ]
        )

    summary = (
        working.groupby("month")[component_cols]
        .mean()
        .reset_index()
        .melt(
            id_vars=["month"],
            var_name="component_variable",
            value_name="avg_component_effect",
        )
        .sort_values(["month", "component_variable"])
        .reset_index(drop=True)
    )
    summary["month"] = summary["month"].astype(int)
    summary["month_label"] = summary["month"].astype(str).str.zfill(2)
    summary.insert(0, "base_variable", base_variable)
    summary.insert(0, "dependent_variable", y_col)
    summary.insert(0, "prompt", prompt)
    summary.insert(0, "model_id", model_id)
    return summary


def write_marginal_effect_txt(
    path: Path,
    title: str,
    summary_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if summary_table.empty:
            handle.write("Marjinal etki ozeti bos.\n")
            return

        for _, row in summary_table.iterrows():
            base_variable_label = str(row.get("base_variable", "base_variable"))
            handle.write(
                f"Ay {int(row['month']):>2} | "
                f"avg={float(row['avg_marginal_effect']):>10.6f} | "
                f"medyan={float(row['median_marginal_effect']):>10.6f} | "
                f"p10={float(row['p10_marginal_effect']):>10.6f} | "
                f"p90={float(row['p90_marginal_effect']):>10.6f} | "
                f"pozitif_oran={float(row['positive_share']):>8.3f} | "
                f"avg_{base_variable_label}={float(row['avg_base_variable']):>10.4f}\n"
            )


def write_marginal_effect_component_txt(
    path: Path,
    title: str,
    component_table: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        if component_table.empty:
            handle.write("Marjinal etki bilesen tablosu bos.\n")
            return

        for month, group in component_table.groupby("month", sort=True):
            handle.write(f"Ay {int(month):>2}\n")
            handle.write("-" * 8 + "\n")
            for _, row in group.sort_values("component_variable").iterrows():
                handle.write(
                    f"{row['component_variable']:<24} "
                    f"{float(row['avg_component_effect']):>12.6f}\n"
                )
            handle.write("\n")


def save_monthly_marginal_effect_plot(
    summary_table: pd.DataFrame,
    title: str,
    output_filename: str,
) -> Path | None:
    if summary_table.empty:
        return None

    plot_data = summary_table.sort_values("month").copy()
    x = plot_data["month"].to_numpy(dtype=float)
    y = plot_data["avg_marginal_effect"].to_numpy(dtype=float)
    p10 = plot_data["p10_marginal_effect"].to_numpy(dtype=float)
    p90 = plot_data["p90_marginal_effect"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", linewidth=2, color="darkred")
    ax.fill_between(x, p10, p90, color="salmon", alpha=0.25)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Ay")
    ax.set_ylabel("Ortalama marjinal etki")
    ax.set_xticks(np.arange(1, 13))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output_path = FIGURES_DIR / output_filename
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    return output_path


def write_lasso_file_guide(path: Path) -> None:
    lines = [
        "LASSO / FINAL MODEL DOSYA ACIKLAMALARI",
        "=====================================",
        "",
        "..._lasso_interaction_adaylari.csv",
        "LASSO'ya verilen tum interaction ve polynomial aday havuzunu gosterir.",
        "Burada hangi terimlerin denendigi, interaction mi polynomial mi oldugu, secilip secilmedigi ve LASSO katsayisi gorulur.",
        "",
        "..._lasso_interaction_adaylari.txt",
        "Ayni aday havuzunun daha okunur metin versiyonudur.",
        "",
        "..._lasso_secilen_interactionlar.csv",
        "LASSO'nun sifirdan farkli katsayi vererek sectigi interaction ve polynomial terimleri gosterir.",
        "Final modele hangi ek terimlerin tasindigini anlamak icin ana dosyadir.",
        "",
        "..._lasso_secilen_interactionlar.txt",
        "Secilen terimlerin sade ve hizli okunur metin versiyonudur.",
        "",
        "..._lasso_ozet_metrikleri.csv / .txt",
        "LASSO asamasinin ozet metriklerini verir.",
        "Ornek: kac aday denendi, kac terim secildi, LASSO alpha degeri ne oldu, LASSO'nun R^2 degeri nedir.",
        "",
        "..._lasso_final_ols_regresyon_tablosu.csv",
        "Secilen interaction/polynomial terimler standardize secim asamasindan sonra ham olcekte yeniden fit edilerek kurulan aciklanabilir OLS tablosudur.",
        "Katsayi, standart hata, t-istatistigi ve p-degeri burada yer alir.",
        "",
        "..._lasso_final_ols_regresyon_tablosu.txt",
        "Final OLS tablosunun sade okunur metin versiyonudur.",
        "",
        "..._lasso_final_ols_ozeti.txt",
        "statsmodels OLS summary ciktisidir.",
        "R^2, Adj. R^2, F-istatistigi ve diger klasik ozet bilgiler burada bulunur.",
        "",
        "..._lasso_final_vif_tablosu.csv / .txt",
        "Final OLS modelindeki degiskenler icin VIF degerlerini verir.",
        "Coklu dogrusal baglanti problemini final modelde tekrar kontrol etmek icin kullanilir.",
        "",
        "..._lasso_final_model_diyagnostik.csv / .txt",
        "Final OLS modelinin diyagnostik tablosudur.",
        "Condition number, Durbin-Watson, gozlem sayisi, R^2 gibi degerleri icerir.",
        "",
        "..._lasso_final_ridge_katsayilari.csv / .txt",
        "Ayni final tasarim matrisi kullanilarak Ridge modeli kurulunca elde edilen katsayilari verir.",
        "Ceza terimi altinda katsayilarin nasil daraldigini gormek icin kullanilir.",
        "",
        "..._lasso_final_ridge_diyagnostik.csv / .txt",
        "Ridge modelinin alpha, R^2 ve condition number gibi ozet metriklerini verir.",
        "",
        "..._lasso_final_elastic_net_katsayilari.csv / .txt",
        "Ayni final tasarim matrisiyle kurulan Elastic Net modelinin katsayilarini verir.",
        "Hem L1 hem L2 cezasi altinda degiskenlerin nasil kaldigini gormek icin kullanilir.",
        "",
        "..._lasso_final_elastic_net_diyagnostik.csv / .txt",
        "Elastic Net modelinin alpha, l1_ratio, R^2 ve benzeri ozet metriklerini verir.",
        "",
        "..._lasso_final_tasarim_degisenleri.csv",
        "Final modele hangi ana etkilerin, interaction'larin ve polynomial terimlerin girdigini listeler.",
        "Final modelin tam degisken setini kontrol etmek icin kullanilir.",
        "",
        "..._lasso_final_cdd_marjinal_etki_aylik.csv / .txt",
        "Final modelden turetilen CDD marjinal etkisinin ay bazinda ozetini verir.",
        "CDD etkisinin hangi aylarda pozitif ya da zayif oldugunu gormek icin kullanilir.",
        "",
        "..._lasso_final_hdd_marjinal_etki_aylik.csv / .txt",
        "Final modelden turetilen HDD marjinal etkisinin ay bazinda ozetini verir.",
        "HDD etkisinin hangi aylarda guclu ya da zayif oldugunu gormek icin kullanilir.",
        "",
        "..._lasso_final_cdd_marjinal_etki_bilesenleri_aylik.csv / .txt",
        "CDD marjinal etkinin OLS tablosundaki hangi katsayilardan geldigini ay bazinda ayristirir.",
        "Ornek: CDD, CDD_sq, CDD_x_month ve toplam etki katkisi birlikte gorulur.",
        "",
        "..._lasso_final_hdd_marjinal_etki_bilesenleri_aylik.csv / .txt",
        "HDD marjinal etkinin OLS tablosundaki hangi katsayilardan geldigini ay bazinda ayristirir.",
        "Ornek: HDD, HDD_sq, HDD_x_month ve toplam etki katkisi birlikte gorulur.",
        "",
        "Pratik okuma sirasi",
        "1. once ..._lasso_secilen_interactionlar.txt dosyasina bak",
        "2. sonra ..._lasso_final_ols_regresyon_tablosu.txt dosyasina bak",
        "3. sonra ..._lasso_final_cdd_marjinal_etki_bilesenleri_aylik.txt ile OLS katsayi katkilarini kontrol et",
        "4. sonra ..._lasso_final_hdd_marjinal_etki_bilesenleri_aylik.txt ile HDD katkilarini kontrol et",
        "5. sonra ..._lasso_final_cdd_marjinal_etki_aylik.txt ile toplam CDD etkisini ay bazinda kontrol et",
        "6. sonra ..._lasso_final_hdd_marjinal_etki_aylik.txt ile toplam HDD etkisini ay bazinda kontrol et",
        "7. sonra ..._lasso_final_vif_tablosu.txt ve ..._lasso_final_model_diyagnostik.txt ile modeli kontrol et",
        "8. en son Ridge ve Elastic Net dosyalariyla saglamlik kontrolu yap",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_year_model_metric_pivot(
    metric_frame: pd.DataFrame,
    metric_name: str,
    report_filename: str,
) -> pd.DataFrame:
    metric_data = metric_frame[metric_frame["metric"] == metric_name].copy()
    if metric_data.empty:
        return pd.DataFrame()

    table_no = pd.to_numeric(
        metric_data["model_id"].astype(str).str.extract(r"model_(\d+)")[0],
        errors="coerce",
    )
    metric_data = metric_data.assign(table_no=table_no).dropna(subset=["table_no"]).copy()
    if metric_data.empty:
        return pd.DataFrame()

    metric_data["table_no"] = metric_data["table_no"].astype(int)
    metric_data["label"] = (
        "Table "
        + metric_data["table_no"].astype(str)
        + "\n"
        + metric_data["prompt"].astype(str).str.replace(r"^TABLE \d+ - ", "", regex=True)
    )
    metric_data["value"] = pd.to_numeric(metric_data["value"], errors="coerce")

    pivot = (
        metric_data.sort_values(["table_no", "year"])
        .pivot(index="label", columns="year", values="value")
        .sort_index()
    )
    pivot.to_csv(REPORTS_DIR / report_filename, encoding="utf-8-sig")
    return pivot


def save_year_model_heatmap(
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    value_label: str,
    output_filename: str,
) -> Path | None:
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(22, 9))
    heatmap_data = pivot.values
    vmax = np.nanmax(heatmap_data)
    if np.isnan(vmax):
        vmax = 1.0
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(1.0, float(vmax)),
    )
    ax.set_title(title, fontsize=16, pad=16)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], fontsize=11)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Yil", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            if pd.isna(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_label, rotation=90)
    plt.tight_layout()

    heatmap_path = FIGURES_DIR / output_filename
    plt.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close()
    return heatmap_path


def save_r_squared_comparison_plot(all_diagnostics: pd.DataFrame) -> list[Path]:
    plot_paths: list[Path] = []

    pivot = build_year_model_metric_pivot(
        all_diagnostics,
        metric_name="r_squared",
        report_filename="r_squared_karsilastirma_tablosu.csv",
    )
    if pivot.empty:
        return plot_paths

    years = [year for year in TARGET_YEARS if year in pivot.columns]
    positions = np.arange(len(pivot.index))
    width = 0.24
    colors = {2022: "#1f77b4", 2023: "#ff7f0e", 2024: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(22, 9))
    for idx, year in enumerate(years):
        offset = (idx - (len(years) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            pivot[year].values,
            width=width,
            label=f"{year} R^2",
            color=colors.get(year, None),
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("2022-2023-2024 OLS Modelleri R^2 Karsilastirmasi", fontsize=16, pad=16)
    ax.set_ylabel("R^2", fontsize=12)
    ax.set_xlabel("OLS Modeli", fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(pivot.index, rotation=0, ha="center", fontsize=9)
    ax.set_ylim(0, max(1.0, float(np.nanmax(pivot.values)) + 0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Yil", frameon=True)
    plt.tight_layout()

    grouped_path = FIGURES_DIR / "r_squared_karsilastirma_grafigi.png"
    plt.savefig(grouped_path, dpi=180, bbox_inches="tight")
    plt.close()
    plot_paths.append(grouped_path)

    heatmap_path = save_year_model_heatmap(
        pivot,
        title="R^2 Isi Haritasi - Hangi Model Hangi Yilda Daha Guclu",
        ylabel="OLS Modeli",
        value_label="R^2",
        output_filename="r_squared_karsilastirma_isiharitasi.png",
    )
    if heatmap_path is not None:
        plot_paths.append(heatmap_path)

    return plot_paths

    fig, ax = plt.subplots(figsize=(22, 9))
    heatmap_data = pivot.values
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1.0, np.nanmax(heatmap_data)))
    ax.set_title("R^2 Isı Haritasi - Hangi Model Hangi Yilda Daha Guclu", fontsize=16, pad=16)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], fontsize=11)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Yil", fontsize=12)
    ax.set_ylabel("OLS Modeli", fontsize=12)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("R^2", rotation=90)
    plt.tight_layout()

    heatmap_path = FIGURES_DIR / "r_squared_karsilastirma_isiharitasi.png"
    plt.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close()
    plot_paths.append(heatmap_path)

    return plot_paths


def save_lasso_r_squared_heatmap(all_lasso_meta: pd.DataFrame) -> list[Path]:
    plot_paths: list[Path] = []
    pivot = build_year_model_metric_pivot(
        all_lasso_meta,
        metric_name="lasso_r_squared",
        report_filename="lasso_r_squared_karsilastirma_tablosu.csv",
    )
    if pivot.empty:
        return plot_paths

    heatmap_path = save_year_model_heatmap(
        pivot,
        title="LASSO R^2 Isi Haritasi - Hangi Model Hangi Yilda Daha Guclu",
        ylabel="LASSO Modeli",
        value_label="LASSO R^2",
        output_filename="lasso_r_squared_karsilastirma_isiharitasi.png",
    )
    if heatmap_path is not None:
        plot_paths.append(heatmap_path)

    return plot_paths


def main() -> None:
    ensure_directories()
    validate_lasso_dependency()
    write_lasso_file_guide(REPORTS_DIR / "lasso_dosya_aciklamalari.txt")

    df = build_analysis_frame()
    all_ols: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    all_scaler: list[pd.DataFrame] = []
    all_vif: list[pd.DataFrame] = []
    all_lasso_candidates: list[pd.DataFrame] = []
    all_lasso_selected: list[pd.DataFrame] = []
    all_lasso_meta: list[pd.DataFrame] = []
    all_cdd_marginal_effects: list[pd.DataFrame] = []
    all_hdd_marginal_effects: list[pd.DataFrame] = []
    all_plot_paths: list[Path] = []

    for year in TARGET_YEARS:
        year_df = df[df["year"] == year].copy()
        year_df.to_csv(PROCESSED_DIR / f"analiz_verisi_{year}.csv", index=False)

        combined_ols: list[pd.DataFrame] = []
        combined_diag: list[pd.DataFrame] = []
        combined_scaler: list[pd.DataFrame] = []
        combined_vif: list[pd.DataFrame] = []
        combined_lasso_candidates: list[pd.DataFrame] = []
        combined_lasso_selected: list[pd.DataFrame] = []
        combined_lasso_meta: list[pd.DataFrame] = []
        combined_cdd_marginal_effects: list[pd.DataFrame] = []
        combined_hdd_marginal_effects: list[pd.DataFrame] = []
        summary_lines: list[str] = []

        for spec in build_model_specs():
            model, fitted_df, ols_table, diagnostics, scaler_table, vif_table = fit_single_model(
                year_df,
                spec,
            )

            model_id = str(spec["model_id"])
            prompt = str(spec["prompt"])
            slug = model_id.replace("model_", "table_")
            prefix = f"{year}_{slug}"

            fitted_df.to_csv(PROCESSED_DIR / f"{prefix}_model_verisi_tahminli.csv", index=False)
            ols_table.to_csv(REPORTS_DIR / f"{prefix}_ols_regresyon_tablosu.csv", index=False)
            diagnostics.to_csv(REPORTS_DIR / f"{prefix}_model_diyagnostik.csv", index=False)
            scaler_table.to_csv(REPORTS_DIR / f"{prefix}_standardizasyon_bazlari.csv", index=False)
            vif_table.to_csv(REPORTS_DIR / f"{prefix}_vif_tablosu.csv", index=False)
            write_vif_txt(
                REPORTS_DIR / f"{prefix}_vif_tablosu.txt",
                f"{year} - {prompt} - VIF",
                vif_table,
            )

            with open(REPORTS_DIR / f"{prefix}_ols_ozeti.txt", "w", encoding="utf-8") as handle:
                handle.write(f"{year} - {prompt}\n\n")
                handle.write(model.summary().as_text())

            combined_ols.append(ols_table.assign(year=year))
            combined_diag.append(diagnostics.assign(year=year))
            combined_scaler.append(scaler_table.assign(year=year))
            combined_vif.append(vif_table.assign(year=year))

            lasso_candidates, lasso_selected, lasso_meta = run_lasso_interaction_selection(
                year_df,
                spec,
            )
            write_lasso_candidate_terms_txt(
                REPORTS_DIR / f"{prefix}_lasso_interaction_adaylari.txt",
                f"{year} - {prompt} - LASSO Aday Interaction ve Polynomial Termler",
                lasso_candidates,
            )
            write_selected_interactions_txt(
                REPORTS_DIR / f"{prefix}_lasso_secilen_interactionlar.txt",
                f"{year} - {prompt} - LASSO Secilen Interaction ve Polynomial Termler",
                lasso_selected,
            )
            write_diagnostics_txt(
                REPORTS_DIR / f"{prefix}_lasso_ozet_metrikleri.txt",
                f"{year} - {prompt} - LASSO Ozet Metrikleri",
                lasso_meta,
            )

            combined_lasso_candidates.append(lasso_candidates.assign(year=year))
            combined_lasso_selected.append(lasso_selected.assign(year=year))
            combined_lasso_meta.append(lasso_meta.assign(year=year))

            lasso_augmented = fit_lasso_augmented_model(year_df, spec, lasso_selected)
            if lasso_augmented is not None:
                (
                    lasso_model,
                    lasso_fitted_df,
                    lasso_ols_table,
                    lasso_diagnostics,
                    lasso_scaler_table,
                    lasso_vif_table,
                ) = lasso_augmented["ols"]
                ridge_coef_table = lasso_augmented["ridge_coef_table"]
                ridge_diagnostics = lasso_augmented["ridge_diagnostics"]
                elastic_net_coef_table = lasso_augmented["elastic_net_coef_table"]
                elastic_net_diagnostics = lasso_augmented["elastic_net_diagnostics"]
                cdd_marginal_effect_summary = lasso_augmented["cdd_marginal_effect_summary"]
                cdd_marginal_effect_component_summary = lasso_augmented["cdd_marginal_effect_component_summary"]
                hdd_marginal_effect_summary = lasso_augmented["hdd_marginal_effect_summary"]
                hdd_marginal_effect_component_summary = lasso_augmented["hdd_marginal_effect_component_summary"]
                design_columns = lasso_augmented["design_columns"]
                lasso_prefix = f"{prefix}_lasso_final"

                lasso_fitted_df.to_csv(
                    PROCESSED_DIR / f"{lasso_prefix}_model_verisi_tahminli.csv",
                    index=False,
                )
                write_ols_table_txt(
                    REPORTS_DIR / f"{lasso_prefix}_ols_regresyon_tablosu.txt",
                    f"{year} - {prompt} - LASSO Final OLS Regresyon Tablosu",
                    lasso_ols_table,
                )
                write_diagnostics_txt(
                    REPORTS_DIR / f"{lasso_prefix}_model_diyagnostik.txt",
                    f"{year} - {prompt} - LASSO Final Model Diyagnostik",
                    lasso_diagnostics,
                )
                lasso_scaler_table.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_standardizasyon_bazlari.csv",
                    index=False,
                )
                write_vif_txt(
                    REPORTS_DIR / f"{lasso_prefix}_vif_tablosu.txt",
                    f"{year} - {prompt} - LASSO Final VIF",
                    lasso_vif_table,
                )
                with open(
                    REPORTS_DIR / f"{lasso_prefix}_ols_ozeti.txt",
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        f"{year} - {prompt} | selection standardized + "
                        "LASSO-selected terms + p-value filtered + raw-scale final OLS\n\n"
                    )
                    handle.write(lasso_model.summary().as_text())
                write_penalized_coefficients_txt(
                    REPORTS_DIR / f"{lasso_prefix}_ridge_katsayilari.txt",
                    f"{year} - {prompt} - LASSO Final Ridge Katsayilari",
                    ridge_coef_table,
                )
                write_diagnostics_txt(
                    REPORTS_DIR / f"{lasso_prefix}_ridge_diyagnostik.txt",
                    f"{year} - {prompt} - LASSO Final Ridge Diyagnostik",
                    ridge_diagnostics,
                )
                write_penalized_coefficients_txt(
                    REPORTS_DIR / f"{lasso_prefix}_elastic_net_katsayilari.txt",
                    f"{year} - {prompt} - LASSO Final Elastic Net Katsayilari",
                    elastic_net_coef_table,
                )
                write_diagnostics_txt(
                    REPORTS_DIR / f"{lasso_prefix}_elastic_net_diyagnostik.txt",
                    f"{year} - {prompt} - LASSO Final Elastic Net Diyagnostik",
                    elastic_net_diagnostics,
                )
                design_columns.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_tasarim_degisenleri.csv",
                    index=False,
                )
                cdd_marginal_effect_summary.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_cdd_marjinal_etki_aylik.csv",
                    index=False,
                )
                cdd_marginal_effect_component_summary.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_cdd_marjinal_etki_bilesenleri_aylik.csv",
                    index=False,
                )
                hdd_marginal_effect_summary.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_hdd_marjinal_etki_aylik.csv",
                    index=False,
                )
                hdd_marginal_effect_component_summary.to_csv(
                    REPORTS_DIR / f"{lasso_prefix}_hdd_marjinal_etki_bilesenleri_aylik.csv",
                    index=False,
                )
                write_marginal_effect_txt(
                    REPORTS_DIR / f"{lasso_prefix}_cdd_marjinal_etki_aylik.txt",
                    f"{year} - {prompt} - CDD Marjinal Etki (Aylik)",
                    cdd_marginal_effect_summary,
                )
                write_marginal_effect_component_txt(
                    REPORTS_DIR / f"{lasso_prefix}_cdd_marjinal_etki_bilesenleri_aylik.txt",
                    f"{year} - {prompt} - CDD Marjinal Etki Bilesenleri (Aylik)",
                    cdd_marginal_effect_component_summary,
                )
                write_marginal_effect_txt(
                    REPORTS_DIR / f"{lasso_prefix}_hdd_marjinal_etki_aylik.txt",
                    f"{year} - {prompt} - HDD Marjinal Etki (Aylik)",
                    hdd_marginal_effect_summary,
                )
                write_marginal_effect_component_txt(
                    REPORTS_DIR / f"{lasso_prefix}_hdd_marjinal_etki_bilesenleri_aylik.txt",
                    f"{year} - {prompt} - HDD Marjinal Etki Bilesenleri (Aylik)",
                    hdd_marginal_effect_component_summary,
                )
                cdd_plot_path = save_monthly_marginal_effect_plot(
                    cdd_marginal_effect_summary,
                    title=f"{year} - {prompt} - Aylik Ortalama CDD Marjinal Etkisi",
                    output_filename=f"{lasso_prefix}_cdd_marjinal_etki_aylik.png",
                )
                if cdd_plot_path is not None:
                    all_plot_paths.append(cdd_plot_path)
                hdd_plot_path = save_monthly_marginal_effect_plot(
                    hdd_marginal_effect_summary,
                    title=f"{year} - {prompt} - Aylik Ortalama HDD Marjinal Etkisi",
                    output_filename=f"{lasso_prefix}_hdd_marjinal_etki_aylik.png",
                )
                if hdd_plot_path is not None:
                    all_plot_paths.append(hdd_plot_path)
                combined_cdd_marginal_effects.append(
                    cdd_marginal_effect_summary.assign(year=year)
                )
                combined_hdd_marginal_effects.append(
                    hdd_marginal_effect_summary.assign(year=year)
                )

            summary_lines.append(f"{year} - {prompt}")
            summary_lines.append(f"OLS table: outputs/reports/{prefix}_ols_regresyon_tablosu.csv")
            summary_lines.append(f"VIF table: outputs/reports/{prefix}_vif_tablosu.csv")
            summary_lines.append(f"OLS summary: outputs/reports/{prefix}_ols_ozeti.txt")
            summary_lines.append(f"LASSO candidates TXT: outputs/reports/{prefix}_lasso_interaction_adaylari.txt")
            summary_lines.append(f"LASSO selected TXT: outputs/reports/{prefix}_lasso_secilen_interactionlar.txt")
            if not lasso_selected.empty:
                summary_lines.append(f"LASSO final OLS TXT: outputs/reports/{prefix}_lasso_final_ols_regresyon_tablosu.txt")
                summary_lines.append(f"LASSO final CDD marginal effect components TXT: outputs/reports/{prefix}_lasso_final_cdd_marjinal_etki_bilesenleri_aylik.txt")
                summary_lines.append(f"LASSO final CDD marginal effect TXT: outputs/reports/{prefix}_lasso_final_cdd_marjinal_etki_aylik.txt")
                summary_lines.append(f"LASSO final HDD marginal effect components TXT: outputs/reports/{prefix}_lasso_final_hdd_marjinal_etki_bilesenleri_aylik.txt")
                summary_lines.append(f"LASSO final HDD marginal effect TXT: outputs/reports/{prefix}_lasso_final_hdd_marjinal_etki_aylik.txt")
                summary_lines.append(f"LASSO final VIF TXT: outputs/reports/{prefix}_lasso_final_vif_tablosu.txt")
                summary_lines.append(f"LASSO final diagnostics TXT: outputs/reports/{prefix}_lasso_final_model_diyagnostik.txt")
                summary_lines.append(f"LASSO final Ridge TXT: outputs/reports/{prefix}_lasso_final_ridge_katsayilari.txt")
                summary_lines.append(f"LASSO final Elastic Net TXT: outputs/reports/{prefix}_lasso_final_elastic_net_katsayilari.txt")
            else:
                summary_lines.append("LASSO final modeller: secilen interaction/polynomial olmadigi icin olusmadi")
            summary_lines.append("")

            if model_id == "model_6":
                all_plot_paths.extend(save_year_plots(fitted_df, year, "fitted_consumption"))

        year_ols = pd.concat(combined_ols, ignore_index=True)
        year_diag = pd.concat(combined_diag, ignore_index=True)
        year_scaler = pd.concat(combined_scaler, ignore_index=True)
        year_vif = pd.concat(combined_vif, ignore_index=True)

        year_ols.to_csv(REPORTS_DIR / f"{year}_tum_ols_regresyon_tablolari.csv", index=False)
        year_diag.to_csv(REPORTS_DIR / f"{year}_tum_model_diyagnostikleri.csv", index=False)
        year_scaler.to_csv(REPORTS_DIR / f"{year}_tum_standardizasyon_bazlari.csv", index=False)
        year_vif.to_csv(REPORTS_DIR / f"{year}_tum_vif_tablolari.csv", index=False)
        year_lasso_candidates = pd.concat(combined_lasso_candidates, ignore_index=True)
        year_lasso_selected = pd.concat(combined_lasso_selected, ignore_index=True)
        year_lasso_meta = pd.concat(combined_lasso_meta, ignore_index=True)
        year_lasso_candidates.to_csv(REPORTS_DIR / f"{year}_tum_lasso_interaction_adaylari.csv", index=False)
        year_lasso_selected.to_csv(REPORTS_DIR / f"{year}_tum_lasso_secilen_interactionlar.csv", index=False)
        year_lasso_meta.to_csv(REPORTS_DIR / f"{year}_tum_lasso_ozet_metrikleri.csv", index=False)
        if combined_cdd_marginal_effects:
            year_cdd_marginal_effects = pd.concat(
                combined_cdd_marginal_effects,
                ignore_index=True,
            )
            year_cdd_marginal_effects.to_csv(
                REPORTS_DIR / f"{year}_tum_lasso_final_cdd_marjinal_etki_aylik.csv",
                index=False,
            )
            all_cdd_marginal_effects.append(year_cdd_marginal_effects)
        if combined_hdd_marginal_effects:
            year_hdd_marginal_effects = pd.concat(
                combined_hdd_marginal_effects,
                ignore_index=True,
            )
            year_hdd_marginal_effects.to_csv(
                REPORTS_DIR / f"{year}_tum_lasso_final_hdd_marjinal_etki_aylik.csv",
                index=False,
            )
            all_hdd_marginal_effects.append(year_hdd_marginal_effects)
        write_grouped_vif_txt(
            REPORTS_DIR / f"{year}_tum_vif_tablolari.txt",
            f"{year} - Tum VIF Tablolari",
            year_vif,
        )
        write_grouped_selected_interactions_txt(
            REPORTS_DIR / f"{year}_tum_lasso_secilen_interactionlar.txt",
            f"{year} - Tum LASSO Secilen Interactionlar",
            year_lasso_selected,
        )

        with open(REPORTS_DIR / f"{year}_model_rehberi.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{year} OLS MODELLERI - DOSYA REHBERI\n\n")
            handle.write("\n".join(summary_lines))

        all_ols.append(year_ols)
        all_diag.append(year_diag)
        all_scaler.append(year_scaler)
        all_vif.append(year_vif)
        all_lasso_candidates.append(year_lasso_candidates)
        all_lasso_selected.append(year_lasso_selected)
        all_lasso_meta.append(year_lasso_meta)

    pd.concat(all_ols, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_ols_regresyon_tablolari.csv",
        index=False,
    )
    pd.concat(all_diag, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_model_diyagnostikleri.csv",
        index=False,
    )
    pd.concat(all_scaler, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_standardizasyon_bazlari.csv",
        index=False,
    )
    pd.concat(all_vif, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_vif_tablolari.csv",
        index=False,
    )
    pd.concat(all_lasso_candidates, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_lasso_interaction_adaylari.csv",
        index=False,
    )
    pd.concat(all_lasso_selected, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_lasso_secilen_interactionlar.csv",
        index=False,
    )
    pd.concat(all_lasso_meta, ignore_index=True).to_csv(
        REPORTS_DIR / "tum_yillar_lasso_ozet_metrikleri.csv",
        index=False,
    )
    if all_cdd_marginal_effects:
        pd.concat(all_cdd_marginal_effects, ignore_index=True).to_csv(
            REPORTS_DIR / "tum_yillar_lasso_final_cdd_marjinal_etki_aylik.csv",
            index=False,
        )
    if all_hdd_marginal_effects:
        pd.concat(all_hdd_marginal_effects, ignore_index=True).to_csv(
            REPORTS_DIR / "tum_yillar_lasso_final_hdd_marjinal_etki_aylik.csv",
            index=False,
        )
    write_grouped_vif_txt(
        REPORTS_DIR / "tum_yillar_vif_tablolari.txt",
        "Tum Yillar - Tum VIF Tablolari",
        pd.concat(all_vif, ignore_index=True),
    )
    write_grouped_selected_interactions_txt(
        REPORTS_DIR / "tum_yillar_lasso_secilen_interactionlar.txt",
        "Tum Yillar - Tum LASSO Secilen Interactionlar",
        pd.concat(all_lasso_selected, ignore_index=True),
    )

    all_diagnostics_df = pd.concat(all_diag, ignore_index=True)
    r2_plot_paths = save_r_squared_comparison_plot(all_diagnostics_df)
    all_lasso_meta_df = pd.concat(all_lasso_meta, ignore_index=True)
    lasso_r2_plot_paths = save_lasso_r_squared_heatmap(all_lasso_meta_df)

    print("2022, 2023 ve 2024 OLS ve LASSO interaction analizleri tamamlandi.")
    for year in TARGET_YEARS:
        print(f"Model rehberi ({year}): {REPORTS_DIR / f'{year}_model_rehberi.txt'}")
        print(f"Toplu OLS tablo ({year}): {REPORTS_DIR / f'{year}_tum_ols_regresyon_tablolari.csv'}")
        print(f"Toplu VIF tablo ({year}): {REPORTS_DIR / f'{year}_tum_vif_tablolari.csv'}")
        print(f"Toplu LASSO interaction secimi ({year}): {REPORTS_DIR / f'{year}_tum_lasso_secilen_interactionlar.csv'}")
    for path in all_plot_paths:
        print(f"Figur: {path}")
    for path in r2_plot_paths:
        print(f"R^2 Figur: {path}")
    for path in lasso_r2_plot_paths:
        print(f"LASSO R^2 Figur: {path}")


if __name__ == "__main__":
    main()
