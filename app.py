import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Intelligent Flight Prediction System",
    page_icon="✈️",
    layout="wide"
)

# ------------------------------------------------
# PROJECT PATH
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")


# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

@st.cache_resource
def load_models():

    delay_model = joblib.load(
        os.path.join(MODELS_DIR, "xgboost_delay.pkl")
    )

    cancellation_model = joblib.load(
        os.path.join(MODELS_DIR, "xgboost_cancellation.pkl")
    )

    severity_model = joblib.load(
        os.path.join(MODELS_DIR, "xgboost_severity.pkl")
    )

    severity_encoder = joblib.load(
        os.path.join(MODELS_DIR, "severity_label_encoder.pkl")
    )

    return (
        delay_model,
        cancellation_model,
        severity_model,
        severity_encoder
    )


@st.cache_data
def load_metadata():

    metadata_path = os.path.join(
        BASE_DIR,
        "app_metadata.json"
    )

    with open(metadata_path, "r") as file:
        return json.load(file)


(
    delay_model,
    cancellation_model,
    severity_model,
    severity_encoder
) = load_models()

metadata = load_metadata()


# ------------------------------------------------
# APPLICATION HEADER
# ------------------------------------------------

st.title("✈️ Intelligent Flight Prediction System")

st.markdown(
    """
    ### Machine Learning Based Flight Prediction

    Predict flight delays, cancellations and delay severity
    using machine learning algorithms.
    """
)

st.divider()


# ------------------------------------------------
# PREDICTION TYPE SELECTION
# ------------------------------------------------

st.subheader("Select Prediction Type")

prediction_type = st.radio(
    "Choose a prediction module:",
    [
        "Flight Delay Prediction",
        "Flight Cancellation Prediction",
        "Flight Delay Severity Prediction"
    ],
    horizontal=True
)

st.divider()


# ------------------------------------------------
# FLIGHT INPUT FORM
# ------------------------------------------------

st.subheader("🛫 Enter Flight Information")

col1, col2 = st.columns(2)

with col1:

    airline = st.selectbox(
        "Select Airline",
        metadata["airlines"]
    )

    origin = st.selectbox(
        "Origin Airport",
        metadata["origins"]
    )

    flight_date = st.date_input(
        "Flight Date",
        value=pd.to_datetime(metadata["max_date"]).date(),
        min_value=pd.to_datetime(metadata["min_date"]).date(),
        max_value=pd.to_datetime(metadata["max_date"]).date()
    )

    departure_time = st.time_input(
        "Scheduled Departure Time"
    )


with col2:

    destination = st.selectbox(
        "Destination Airport",
        metadata["destinations"]
    )

    distance = st.number_input(
        "Flight Distance (miles)",
        min_value=1.0,
        max_value=10000.0,
        value=500.0
    )

    duration = st.number_input(
        "Scheduled Flight Duration (minutes)",
        min_value=1.0,
        max_value=1500.0,
        value=120.0
    )


st.divider()


# ------------------------------------------------
# CREATE INPUT DATA
# ------------------------------------------------

input_data = pd.DataFrame([{

    "AIRLINE": airline,

    "ORIGIN": origin,

    "DEST": destination,

    "YEAR": flight_date.year,

    "MONTH": flight_date.month,

    "DAY": flight_date.day,

    "DAY_OF_WEEK": flight_date.weekday(),

    "DEP_HOUR": departure_time.hour,

    "DISTANCE": distance,

    "CRS_ELAPSED_TIME": duration

}])


# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------

button_label = {
    "Flight Delay Prediction": "Predict Flight Delay",
    "Flight Cancellation Prediction": "Predict Flight Cancellation",
    "Flight Delay Severity Prediction": "Predict Delay Severity"
}

if st.button(
    button_label[prediction_type],
    use_container_width=True
):

    if origin == destination:

        st.error(
            "Origin and destination airports must be different."
        )

    else:

        st.divider()

        st.subheader("📊 Flight Prediction Result")


        # ============================================
        # 1. FLIGHT DELAY PREDICTION
        # ============================================

        if prediction_type == "Flight Delay Prediction":

            prediction = delay_model.predict(input_data)[0]

            probabilities = delay_model.predict_proba(input_data)[0]

            classes = delay_model.named_steps[
                "classifier"
            ].classes_

            delay_class_index = list(classes).index(1)

            probability = probabilities[delay_class_index]

            if prediction == 1:

                st.error(
                    "⚠️ Flight is predicted to be DELAYED"
                )

            else:

                st.success(
                    "✅ Flight is predicted to be NOT DELAYED"
                )

            st.metric(
                "Estimated Delay Score",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.info(
                """
                The model estimates whether the flight's
                departure delay will be at least 15 minutes.
                This is not a guarantee of actual flight status.
                """
            )


        # ============================================
        # 2. FLIGHT CANCELLATION PREDICTION
        # ============================================

        elif prediction_type == "Flight Cancellation Prediction":

            prediction = cancellation_model.predict(
                input_data
            )[0]

            probabilities = cancellation_model.predict_proba(
                input_data
            )[0]

            classes = cancellation_model.named_steps[
                "classifier"
            ].classes_

            cancellation_class_index = list(classes).index(1)

            probability = probabilities[cancellation_class_index]

            if prediction == 1:

                st.error(
                    "⚠️ Flight is predicted to be CANCELLED"
                )

            else:

                st.success(
                    "✅ Flight is predicted to be NOT CANCELLED"
                )

            st.metric(
                "Estimated Cancellation Score",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            st.warning(
                """
                Cancellation prediction has limited precision
                in the model evaluation. A high or low score
                should not be treated as confirmation of flight
                cancellation. Verify actual flight status
                with the airline.
                """
            )


        # ============================================
        # 3. FLIGHT DELAY SEVERITY PREDICTION
        # ============================================

        elif prediction_type == "Flight Delay Severity Prediction":

            prediction_encoded = severity_model.predict(
                input_data
            )[0]

            prediction_label = severity_encoder.inverse_transform(
                np.array([int(prediction_encoded)])
            )[0]

            probabilities = severity_model.predict_proba(
                input_data
            )[0]

            classifier_classes = severity_model.named_steps[
                "classifier"
            ].classes_

            predicted_class_index = list(
                classifier_classes
            ).index(int(prediction_encoded))

            probability = probabilities[predicted_class_index]

            severity_ranges = {
                "On-Time": "0–14 minutes",
                "Minor": "15–30 minutes",
                "Moderate": "31–60 minutes",
                "Major": "61–120 minutes",
                "Severe": "More than 120 minutes"
            }

            if prediction_label == "On-Time":

                st.success(
                    "✅ Predicted Delay Severity: ON-TIME"
                )

            elif prediction_label == "Minor":

                st.info(
                    "🟡 Predicted Delay Severity: MINOR"
                )

            elif prediction_label == "Moderate":

                st.warning(
                    "🟠 Predicted Delay Severity: MODERATE"
                )

            elif prediction_label == "Major":

                st.error(
                    "🔴 Predicted Delay Severity: MAJOR"
                )

            else:

                st.error(
                    "🚨 Predicted Delay Severity: SEVERE"
                )

            st.metric(
                "Predicted Delay Category",
                prediction_label
            )

            st.write(
                "**Expected Delay Range:**",
                severity_ranges[prediction_label]
            )

            st.metric(
                "Model Score for Predicted Category",
                f"{probability * 100:.2f}%"
            )

            st.info(
                """
                The model predicts one of five departure-delay
                severity categories. Its performance varies
                across categories, so this prediction is an
                estimate rather than a guarantee.
                """
            )


        # ============================================
        # DISPLAY INPUT DETAILS
        # ============================================

        st.divider()

        st.subheader("Entered Flight Details")

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Developed using Python, Streamlit and XGBoost | "
    "End-to-End Machine Learning College Project"
)