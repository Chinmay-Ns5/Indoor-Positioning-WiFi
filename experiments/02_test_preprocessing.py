from pathlib import Path

from src.data_utils import (
    load_data,
    get_wap_columns,
    get_active_wap_columns,
    split_by_building,
    split_by_building_nan,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "trainingData.csv"
VALIDATION_PATH = ROOT / "data" / "validationData.csv"


def main():

    print("=" * 60)
    print("DATA PREPROCESSING VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    train_df, validation_df = load_data(
        TRAIN_PATH,
        VALIDATION_PATH
    )

    print("\nOriginal shapes:")
    print(f"Training:   {train_df.shape}")
    print(f"Validation: {validation_df.shape}")

    # --------------------------------------------------
    # 2. Identify WAP columns
    # --------------------------------------------------

    wap_cols = get_wap_columns(train_df)

    print("\nWAP features:")
    print(f"Total WAP columns: {len(wap_cols)}")

    # --------------------------------------------------
    # 3. Remove never-heard WAPs
    # --------------------------------------------------

    active_waps = get_active_wap_columns(train_df)

    print("\nWAP filtering:")
    print(f"Original WAP columns: {len(wap_cols)}")
    print(f"Active WAP columns:   {len(active_waps)}")
    print(f"Removed WAP columns:  {len(wap_cols) - len(active_waps)}")

    # --------------------------------------------------
    # 4. Verify 100 -> -105
    # --------------------------------------------------

    X_train, y_train = split_by_building(
        train_df,
        active_waps,
        building_id=0
    )

    print("\nBuilding 0:")
    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    print("\nCleaned RSSI range:")
    print(f"Minimum: {X_train.min().min()}")
    print(f"Maximum: {X_train.max().max()}")

    print("\nRemaining 100 values:")
    print((X_train == 100).sum().sum())

    # --------------------------------------------------
    # 5. Check floor balance
    # --------------------------------------------------

    print("\nBuilding 0 floor distribution:")
    print(y_train.value_counts().sort_index())

    # --------------------------------------------------
    # 6. Check all buildings
    # --------------------------------------------------

    print("\n--- All Buildings ---")

    for building_id in sorted(train_df["BUILDINGID"].unique()):

        X, y = split_by_building(
            train_df,
            active_waps,
            building_id
        )

        print(
            f"Building {building_id}: "
            f"X={X.shape}, "
            f"floors={sorted(y.unique())}"
        )

    # --------------------------------------------------
    # 7. Verify NaN representation
    # --------------------------------------------------

    X_nan, _ = split_by_building_nan(
        train_df,
        active_waps,
        building_id=0
    )

    print("\nNaN representation:")
    print(f"NaN values: {X_nan.isna().sum().sum():,}")

    print("\nPreprocessing verification complete.")


if __name__ == "__main__":
    main()