import pandas as pd
import numpy as np


def load_data(train_path, validation_path):
    """
    Load the UJIndoorLoc training and validation datasets.
    """
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    return train_df, validation_df


def get_wap_columns(df):
    """
    Return all WiFi Access Point (WAP) feature columns.
    """
    return [col for col in df.columns if col.startswith("WAP")]


def get_active_wap_columns(train_df):
    """
    Return WAP columns that are heard at least once
    in the training dataset.

    A WAP containing only 100 means it was never detected
    and therefore carries no useful information.
    """
    wap_cols = get_wap_columns(train_df)

    active_waps = [
        col for col in wap_cols
        if (train_df[col] != 100).any()
    ]

    return active_waps


def clean_rssi(df, wap_cols, fill=-105):
    """
    Replace RSSI value 100, which means 'not detected',
    with a weak signal value.

    Default:
        100 -> -105

    The actual detected RSSI range is -104 to 0.
    Therefore -105 represents a signal weaker than
    the weakest observed real signal.
    """
    X = df[wap_cols].copy()
    X = X.replace(100, fill)

    return X


def clean_rssi_nan(df, wap_cols):
    """
    Replace RSSI value 100 with NaN.

    This representation is used for experiments involving
    models that can explicitly handle missing values.
    """
    X = df[wap_cols].copy()
    X = X.replace(100, np.nan)

    return X


def split_by_building(df, wap_cols, building_id, fill=-105):
    """
    Extract RSSI features and FLOOR labels for one building.

    Returns:
        X : cleaned RSSI features
        y : floor labels
    """
    building_df = df[df["BUILDINGID"] == building_id].copy()

    X = clean_rssi(building_df, wap_cols, fill=fill)
    y = building_df["FLOOR"].copy()

    return X, y


def split_by_building_nan(df, wap_cols, building_id):
    """
    Extract RSSI features and FLOOR labels for one building
    while preserving missing RSSI values as NaN.
    """
    building_df = df[df["BUILDINGID"] == building_id].copy()

    X = clean_rssi_nan(building_df, wap_cols)
    y = building_df["FLOOR"].copy()

    return X, y