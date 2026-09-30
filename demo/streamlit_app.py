import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WiFi Indoor Positioning System",
    page_icon="📍",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp { background: #f3f5f1; color: #1c2924; }
    [data-testid="stSidebar"] { background: #202b27; }
    [data-testid="stSidebar"] * { color: #eef3ef; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce4dd; padding: 14px 16px; border-radius: 6px; }
    [data-testid="stMetricLabel"] { color: #56665e; }
    .stButton > button[kind="primary"] { background: #176b52; border-color: #176b52; }
    .stButton > button[kind="primary"]:hover { background: #10543f; border-color: #10543f; }
    div[data-testid="stAlert"] { border-radius: 5px; }
    h1, h2, h3 { color: #18372c; }
    @media (max-width: 700px) {
        [data-testid="stMetric"] { padding: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = DATA_DIR / "trainingData.csv"
VALIDATION_PATH = DATA_DIR / "validationData.csv"

BUILDING_MODEL_PATH = MODEL_DIR / "building_knn.joblib"

FLOOR_MODEL_PATHS = {
    0: MODEL_DIR / "floor_building_0.joblib",
    1: MODEL_DIR / "floor_building_1.joblib",
    2: MODEL_DIR / "floor_building_2.joblib",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path):
    """Check whether a required file exists."""
    return path.exists()


def load_joblib(path):
    """Load a joblib model."""
    return joblib.load(path)


def get_classifier(obj):
    """
    Some saved objects may be classifiers directly.
    Others may be dictionaries containing the classifier.

    This function safely extracts the actual estimator.
    """

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):

        possible_keys = [
            "model",
            "classifier",
            "estimator",
            "clf",
            "building_model",
            "floor_model",
        ]

        for key in possible_keys:
            value = obj.get(key)

            if value is not None and hasattr(value, "predict"):
                return value

        # Search all dictionary values
        for value in obj.values():
            if hasattr(value, "predict"):
                return value

    raise TypeError(
        f"Could not find a prediction model inside object of type "
        f"{type(obj).__name__}"
    )


def get_feature_names(model):
    """
    Get the exact feature names used when the model was trained.

    This is important because the project uses 465 active WAP
    features rather than all 520 WAP columns.
    """

    if isinstance(model, dict):
        for key in ["wap_columns", "feature_names", "features", "columns", "feature_columns"]:
            value = model.get(key)
            if value is not None and len(value) > 0:
                return list(value)

    classifier = get_classifier(model)

    if hasattr(classifier, "feature_names_in_"):
        return list(classifier.feature_names_in_)

    return None


def get_model_scaler(model):
    if isinstance(model, dict):
        return model.get("scaler")
    return None


def get_missing_value(model):
    if isinstance(model, dict):
        return model.get("missing_fill", -105)
    return -105


def prepare_fingerprint(row, feature_names, missing_value=-105):
    """
    Convert one validation row into the exact feature matrix
    expected by the trained model.
    """

    if feature_names is None:
        raise ValueError(
            "The saved model does not contain feature_names_in_. "
            "Cannot safely determine the exact WAP feature order."
        )

    missing = [
        feature
        for feature in feature_names
        if feature not in row.index
    ]

    if missing:
        raise ValueError(
            f"Validation data is missing {len(missing)} model features."
        )

    X = row[feature_names].copy()

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # UJIndoorLoc uses 100 for an undetected access point.
    X = X.replace(100, missing_value).fillna(missing_value)

    # Keep the exact DataFrame column names.
    # This prevents feature-order/name problems.
    return X.to_frame().T


def predict_model(model, X):
    """
    Safely call the classifier.
    """

    classifier = get_classifier(model)

    scaler = get_model_scaler(model)
    model_input = scaler.transform(X) if scaler is not None else X
    prediction = classifier.predict(model_input)

    return prediction


def prediction_confidence(model, X):
    """
    Return probability/confidence when available.
    """

    classifier = get_classifier(model)

    scaler = get_model_scaler(model)
    model_input = scaler.transform(X) if scaler is not None else X

    if hasattr(classifier, "predict_proba"):

        probabilities = classifier.predict_proba(model_input)

        if probabilities is not None and len(probabilities) > 0:
            return float(np.max(probabilities[0]))

    return None


def normalize_label(value):
    """
    Convert numpy scalar values to normal Python values.
    """

    try:
        return int(value)
    except Exception:
        return value


def get_actual_building(row):
    possible_names = [
        "BUILDINGID",
        "BuildingID",
        "building",
        "building_id",
    ]

    for name in possible_names:
        if name in row.index:
            return int(row[name])

    return None


def get_actual_floor(row):
    possible_names = [
        "FLOOR",
        "Floor",
        "floor",
        "floor_id",
    ]

    for name in possible_names:
        if name in row.index:
            return int(row[name])

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found:\n{VALIDATION_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    return train_df, validation_df


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not BUILDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Building model not found:\n{BUILDING_MODEL_PATH}"
        )

    building_model = load_joblib(BUILDING_MODEL_PATH)

    floor_models = {}

    for building_id, path in FLOOR_MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Floor model for Building {building_id} not found:\n{path}"
            )

        floor_models[building_id] = load_joblib(path)

    return building_model, floor_models


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

try:

    train_df, validation_df = load_data()

    building_model, floor_models = load_models()

    building_features = get_feature_names(building_model)

    if building_features is None:
        raise ValueError(
            "Could not determine the WAP features used by the "
            "building model."
        )

    system_ready = True
    initialization_error = None

except Exception as e:

    system_ready = False
    initialization_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System status")

    if system_ready:

        st.success("Models and validation data ready")
        st.write(f"**Validation fingerprints:** {len(validation_df):,}")
        st.write(f"**WAP features per model:** {len(building_features)}")
        st.write(f"**Floor models:** {len(floor_models)} building-specific models")
        st.caption("Stage 1: kNN building classification. Stage 2: Gradient Boosting floor classification.")

    else:

        st.error("System could not start")
        st.code(initialization_error)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("WiFi Indoor Positioning")
st.caption("Explore the UJIndoorLoc validation fingerprints and predict a building and floor from each WiFi RSSI sample.")


# ============================================================
# STOP IF SYSTEM FAILED
# ============================================================

if not system_ready:

    st.warning(
        "The UI loaded, but the ML system could not be initialized. "
        "Check the error shown in the sidebar."
    )

    st.stop()


# ============================================================
# WIFI FINGERPRINT SECTION
# ============================================================

st.subheader("Choose a validation fingerprint")
st.write("Each sample contains received signal strength (RSSI) values from nearby WiFi access points.")


sample_index = st.number_input(
    "Sample number",
    min_value=0,
    max_value=len(validation_df) - 1,
    value=0,
    step=1,
)


# ============================================================
# SAMPLE INFORMATION
# ============================================================

selected_row = validation_df.iloc[int(sample_index)]

actual_building = get_actual_building(selected_row)
actual_floor = get_actual_floor(selected_row)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected sample",
        f"#{int(sample_index):,}"
    )

with col2:
    if actual_building is not None:
        st.metric(
            "Reference building",
            f"Building {actual_building}"
        )
    else:
        st.metric(
            "Actual Building",
            "Unknown"
        )

with col3:
    if actual_floor is not None:
        st.metric(
            "Reference floor",
            f"Floor {actual_floor}"
        )
    else:
        st.metric(
            "Actual Floor",
            "Unknown"
        )

with st.expander("Inspect this WiFi fingerprint"):
    signal_values = pd.to_numeric(
        selected_row[building_features], errors="coerce"
    ).replace([100, np.inf, -np.inf], np.nan).dropna()
    strongest_signals = signal_values.sort_values(ascending=False).head(12)
    signal_col, data_col = st.columns([1, 2])
    with signal_col:
        st.metric("Detected access points", f"{len(signal_values)} / {len(building_features)}")
        st.caption("Undetected access points are stored as 100 and treated as missing RSSI.")
    with data_col:
        st.caption("Strongest detected signals")
        if not strongest_signals.empty:
            signal_chart = pd.DataFrame({
                "Access point": strongest_signals.index,
                "RSSI (dBm)": strongest_signals.values,
            })
            signal_table = signal_chart.style.bar(
                subset=["RSSI (dBm)"], color="#176b52", align="mid"
            )
            st.dataframe(signal_table, hide_index=True, use_container_width=True)
        else:
            st.info("This sample contains no detected access points.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_location(row):

    # --------------------------------------------------------
    # Prepare building features
    # --------------------------------------------------------

    X_building = prepare_fingerprint(
        row,
        building_features,
        get_missing_value(building_model),
    )

    # --------------------------------------------------------
    # STAGE 1 - BUILDING
    # --------------------------------------------------------

    building_prediction = predict_model(
        building_model,
        X_building
    )

    building_prediction = normalize_label(
        building_prediction[0]
    )

    building_confidence = prediction_confidence(
        building_model,
        X_building
    )

    # --------------------------------------------------------
    # STAGE 2 - FLOOR
    # --------------------------------------------------------

    if building_prediction not in floor_models:

        raise ValueError(
            f"No floor model available for Building "
            f"{building_prediction}."
        )

    floor_model = floor_models[building_prediction]

    floor_features = get_feature_names(floor_model)

    if floor_features is None:
        raise ValueError(
            f"The floor model for Building {building_prediction} "
            "does not contain its expected WAP feature list."
        )

    X_floor = prepare_fingerprint(
        row,
        floor_features,
        get_missing_value(floor_model),
    )

    floor_prediction = predict_model(
        floor_model,
        X_floor
    )

    floor_prediction = normalize_label(
        floor_prediction[0]
    )

    floor_confidence = prediction_confidence(
        floor_model,
        X_floor
    )

    return {
        "building": building_prediction,
        "floor": floor_prediction,
        "building_confidence": building_confidence,
        "floor_confidence": floor_confidence,
    }


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "📍 Predict Location",
    type="primary",
    use_container_width=True,
):

    try:

        result = predict_location(selected_row)

        predicted_building = result["building"]
        predicted_floor = result["floor"]

        building_confidence = result[
            "building_confidence"
        ]

        floor_confidence = result[
            "floor_confidence"
        ]

        # ----------------------------------------------------
        # CORRECTNESS
        # ----------------------------------------------------

        building_correct = (
            actual_building is not None
            and predicted_building == actual_building
        )

        floor_correct = (
            actual_floor is not None
            and predicted_floor == actual_floor
        )

        end_to_end_correct = (
            building_correct and floor_correct
        )

        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.info(
            f"Model prediction: Building {predicted_building}, "
            f"Floor {predicted_floor}"
        )

        st.subheader("Prediction details")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Predicted building", predicted_building)

        with c2:
            st.metric("Predicted floor", predicted_floor)

        with c3:

            if building_confidence is not None:

                confidence_text = (
                    f"{building_confidence:.2%}"
                )

            else:

                confidence_text = "N/A"

            st.metric("Building model score", confidence_text)

        with c4:

            if floor_confidence is not None:

                confidence_text = (
                    f"{floor_confidence:.2%}"
                )

            else:

                confidence_text = "N/A"

            st.metric("Floor model score", confidence_text)

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        st.subheader("Actual vs predicted location")

        validation_col1, validation_col2 = st.columns(2)

        with validation_col1:
            actual_location = (
                f"Building {actual_building}, Floor {actual_floor}"
                if actual_building is not None and actual_floor is not None
                else "Reference unavailable"
            )
            st.metric("Actual location", actual_location)

        with validation_col2:
            st.metric(
                "Predicted location",
                f"Building {predicted_building}, Floor {predicted_floor}",
            )

        st.caption("The building model selects which building-specific floor model runs next.")
        st.markdown("**Component checks**")
        check_col1, check_col2 = st.columns(2)

        with check_col1:
            if building_correct:
                st.success("Building: CORRECT")
            else:
                st.error("Building: INCORRECT")

        with check_col2:
            if floor_correct:
                st.success("Floor: CORRECT")
            else:
                st.error("Floor: INCORRECT")

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        if end_to_end_correct:
            st.success("End-to-end result: CORRECT")

        else:
            st.error("End-to-end result: INCORRECT")

        st.caption("Model scores are maximum class probabilities when available; they are not calibrated guarantees of accuracy.")

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.divider()

st.subheader("How this connects to the report")
with st.expander("Research findings and this demo", expanded=False):
    st.markdown(
        """
        The report compares kNN, SVM, decision trees, and Gradient Boosting, and studies PCA feature counts, training sample size, and missing WiFi signals.

        - kNN performed strongly as the training sample grew; adding many more PCA features gave smaller gains.
        - SVM performed less well on this high-dimensional fingerprint task.
        - Gradient Boosting improved tree-based accuracy and was reported as robust to missing signals.
        - This app runs the saved two-stage model: kNN predicts the building, then that building's Gradient Boosting model predicts the floor.

        The selected validation sample has known building and floor labels, so this screen compares a model prediction against its reference label. The supplied report's cross-validation experiments are broader than this single-sample demo.
        """
    )

st.subheader("Prediction pipeline")

architecture_col1, architecture_col2 = st.columns(2)

with architecture_col1:

    st.markdown(
        """
        ### 1. Predict building

        **WiFi RSSI fingerprint**

        ↓

        **k-Nearest Neighbors**

        ↓

        **Building prediction**
        """
    )

with architecture_col2:

    st.markdown(
        """
        ### 2. Predict floor

        **Same WiFi fingerprint**

        ↓

        **Building-specific Gradient Boosting model**

        ↓

        **Floor prediction**
        """
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("Loaded Models")

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.write(
        "**Building model:** "
        "K-Nearest Neighbors"
    )

    st.write(
        f"**Input features:** "
        f"{len(building_features)} active WAP features"
    )

with model_col2:

    st.write(
        "**Floor models:** "
        "Gradient Boosting"
    )

    st.write(
        "**Models available:** Building 0, "
        "Building 1, Building 2"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "WiFi Indoor Positioning System — "
    "Two-stage RSSI fingerprint localization"
)