
import streamlit as st
import pandas as pd
import joblib
import json

MODEL_FILENAME = "rf_model.pkl"
FEATURES_FILENAME = "feature_columns.json"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILENAME)
    with open(FEATURES_FILENAME, "r") as f:
        feature_info = json.load(f)
    return model, feature_info

model, feature_info = load_artifacts()

st.title("Food Delivery Customer Churn Prediction")

st.write("Fill in the details below and click **Predict Churn**.")

categorical_cols = feature_info["categorical_cols"]
numeric_cols = feature_info["numeric_cols"]
categories = feature_info["categories"]
input_cols = feature_info["input_columns"]

input_values = {}

with st.form("input_form"):
    for col in input_cols:
        if col in categorical_cols:
            opts = categories.get(col, [])
            if len(opts) > 0:
                val = st.selectbox(col, opts)
            else:
                val = st.text_input(col)
            input_values[col] = val
        elif col in numeric_cols:
            val = st.number_input(col, value=0.0)
            input_values[col] = val
        else:
            val = st.text_input(col)
            input_values[col] = val

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([input_values])

    try:
        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error during prediction: {repr(e)}")
    else:
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: **{prob:.2f}**")
        st.write(f"Predicted Class: **{'Churned' if pred == 1 else 'Active'}**")
