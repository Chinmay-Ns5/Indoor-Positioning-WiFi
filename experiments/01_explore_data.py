import pandas as pd
from pathlib import Path


# Project root directory
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def main():
    print("=" * 60)
    print("UJIndoorLoc Dataset Exploration")
    print("=" * 60)

    # Load datasets
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    print("\n--- Dataset Shapes ---")
    print(f"Training data:   {train.shape}")
    print(f"Validation data: {validation.shape}")

    # Identify WAP columns
    wap_cols = [col for col in train.columns if col.startswith("WAP")]

    print("\n--- WiFi Features ---")
    print(f"Number of WAP features: {len(wap_cols)}")
    print(f"First WAP column: {wap_cols[0]}")
    print(f"Last WAP column:  {wap_cols[-1]}")

    # Buildings
    print("\n--- Buildings ---")
    print(train["BUILDINGID"].value_counts().sort_index())

    # Floors by building
    print("\n--- Floors by Building ---")
    print(
        train.groupby("BUILDINGID")["FLOOR"]
        .value_counts()
        .sort_index()
    )

    # Missing signal value
    print("\n--- RSSI Value Analysis ---")
    wap_data = train[wap_cols]

    total_values = wap_data.size
    not_detected = (wap_data == 100).sum().sum()

    print(f"Total RSSI values: {total_values:,}")
    print(f"Values equal to 100: {not_detected:,}")
    print(
        f"Percentage equal to 100: "
        f"{100 * not_detected / total_values:.2f}%"
    )

    # RSSI range excluding 100
    detected_values = wap_data[wap_data != 100]

    print("\nDetected RSSI range:")
    print(f"Minimum: {detected_values.min().min()}")
    print(f"Maximum: {detected_values.max().max()}")

    # Dataset columns
    print("\n--- Important Columns ---")
    print(train.columns.tolist())

    print("\n--- First 5 Rows ---")
    print(train.head())

    print("\nExploration complete.")


if __name__ == "__main__":
    main()