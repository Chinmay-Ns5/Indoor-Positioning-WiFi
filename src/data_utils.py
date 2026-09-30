import pandas as pd
import numpy as np


def load_data(train_path, validation_path):
    """
    Load training and validation datasets from CSV files.

    Args:
        train_path: Path to the training CSV file.
        validation_path: Path to the validation CSV file.

    Returns:
        Tuple containing the training and validation DataFrames.
    """
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    return train_df, validation_df


def get_wap_columns(df):
    """
    Get all WiFi Access Point (WAP) feature columns.

    Args:
        df: Input dataset containing WAP columns.

    Returns:
        List of column names starting with 'WAP'.
    """
    return [
        col for col in df.columns
        if col.startswith("WAP")
    ]


def get_active_wap_columns(train_df):
    """
    Get WAP columns containing at least one detected signal.

    Args:
        train_df: Training DataFrame containing WAP features.

    Returns:
        List of WAP columns that contain at least one value other than 100.
    """
    wap_cols = get_wap_columns(train_df)

    active_waps = [
        col
        for col in wap_cols
        if (train_df[col] != 100).any()
    ]

    return active_waps


def clean_rssi(df, wap_cols, fill=-105):
    """
    Replace undetected RSSI values with a weak signal value.

    Args:
        df: Input DataFrame containing RSSI features.
        wap_cols: List of WAP feature column names.
        fill: Value used to replace undetected signals.

    Returns:
        DataFrame containing cleaned RSSI features.
    """
    X = df[wap_cols].copy()
    X = X.replace(100, fill)

    return X


def clean_rssi_nan(df, wap_cols):
    """
    Replace undetected RSSI values with NaN.

    Args:
        df: Input DataFrame containing RSSI features.
        wap_cols: List of WAP feature column names.

    Returns:
        DataFrame containing RSSI features with missing values as NaN.
    """
    X = df[wap_cols].copy()
    X = X.replace(100, np.nan)

    return X


def split_by_building(df, wap_cols, building_id, fill=-105):
    """
    Extract cleaned RSSI features and floor labels for one building.

    Args:
        df: Complete dataset.
        wap_cols: List of WAP feature column names.
        building_id: Building identifier to select.
        fill: Value used to replace undetected RSSI signals.

    Returns:
        Tuple containing the cleaned RSSI features and floor labels.
    """
    building_df = df[
        df["BUILDINGID"] == building_id
    ].copy()

    X = clean_rssi(
        building_df,
        wap_cols,
        fill=fill
    )

    y = building_df["FLOOR"].copy()

    return X, y


def split_by_building_nan(df, wap_cols, building_id):
    """
    Extract RSSI features and floor labels while preserving missing values.

    Args:
        df: Complete dataset.
        wap_cols: List of WAP feature column names.
        building_id: Building identifier to select.

    Returns:
        Tuple containing RSSI features with NaN values and floor labels.
    """
    building_df = df[
        df["BUILDINGID"] == building_id
    ].copy()

    X = clean_rssi_nan(
        building_df,
        wap_cols
    )

    y = building_df["FLOOR"].copy()

    return X, y