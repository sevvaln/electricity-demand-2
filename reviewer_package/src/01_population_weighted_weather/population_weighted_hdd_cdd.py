from __future__ import annotations

from pathlib import Path

from talep_tahmin_tek_dosya import (
    PROJECT_ROOT,
    build_population_weighted_degree_hour_frame,
)


POPULATION_FILE = Path(r"C:\Users\Monster\Downloads\2022ilbazındanufusverileri.xlsx")
YEARS = [2022, 2023, 2024, 2025]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
MANUAL_CITY_MAPPING = {
    "KARABUK": "KARABÜK",
    "KARABÜK KAPULLU": "KARABÜK",
}


def main() -> None:
    final_series, metadata = build_population_weighted_degree_hour_frame(
        years=YEARS,
        population_path=POPULATION_FILE,
        output_dir=OUTPUT_DIR,
        manual_city_mapping=MANUAL_CITY_MAPPING,
        require_full_target_coverage=False,
    )

    print("Manual city-name mapping dictionary:")
    print(MANUAL_CITY_MAPPING)
    print()
    print(f"Temperature unique cities: {metadata['temperature_unique_city_count']}")
    print(f"Population unique cities: {metadata['population_unique_city_count']}")
    print(f"Matched city count: {metadata['matched_city_count']}")
    print(f"Final hourly series path: {metadata['final_series_path']}")
    print(f"Weights path: {metadata['weights_path']}")
    print(f"Adjusted yearly weights path: {metadata['yearly_adjusted_weights_path']}")
    print(f"Coverage report path: {metadata['coverage_report_path']}")
    print(f"Methodology note path: {metadata['methodology_note_path']}")
    print()
    print("Final hourly series sample:")
    print(final_series.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
