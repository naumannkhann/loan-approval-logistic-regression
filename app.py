
import streamlit as st
import pandas as pd
import joblib

# Load model files only once
@st.cache_resource
def load_model_files():
    model = joblib.load("loan_approval_model.pkl")
    scaler = joblib.load("scaler.pkl")
    education_encoder = joblib.load("education_encoder.pkl")
    self_employed_encoder = joblib.load("self_employed_encoder.pkl")
    loan_status_encoder = joblib.load("loan_status_encoder.pkl")

    return (
        model,
        scaler,
        education_encoder,
        self_employed_encoder,
        loan_status_encoder
    )


(
    model,
    scaler,
    education_encoder,
    self_employed_encoder,
    loan_status_encoder
) = load_model_files()


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter the applicant's details to predict whether the loan "
    "is likely to be approved or rejected."
)


with st.form("loan_form"):

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    education = st.selectbox(
        "Education",
        education_encoder.classes_.tolist()
    )

    self_employed = st.selectbox(
        "Self Employed",
        self_employed_encoder.classes_.tolist()
    )

    income_annum = st.number_input(
        "Annual Income",
        min_value=0,
        value=5000000,
        step=100000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000000,
        step=100000
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=30,
        value=10,
        step=1
    )

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=700,
        step=1
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=3000000,
        step=100000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=2000000,
        step=100000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=5000000,
        step=100000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=1500000,
        step=100000
    )

    submitted = st.form_submit_button("Predict Loan Status")


if submitted:

    try:
        education_encoded = education_encoder.transform(
            [education]
        )[0]

        self_employed_encoded = self_employed_encoder.transform(
            [self_employed]
        )[0]

        input_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education_encoded,
            "self_employed": self_employed_encoded,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])

        # Scale input using the same scaler used during training
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        probabilities = model.predict_proba(input_data_scaled)[0]

        result = loan_status_encoder.inverse_transform(
            prediction
        )[0].strip()

        class_names = [
            str(label).strip()
            for label in loan_status_encoder.classes_
        ]

        approved_index = class_names.index("Approved")
        rejected_index = class_names.index("Rejected")

        approved_probability = (
            probabilities[approved_index] * 100
        )

        rejected_probability = (
            probabilities[rejected_index] * 100
        )

        if result.lower() == "approved":
            st.success(f"Loan Status: {result}")
        else:
            st.error(f"Loan Status: {result}")

        st.write(
            f"Approval Probability: "
            f"{approved_probability:.2f}%"
        )

        st.write(
            f"Rejection Probability: "
            f"{rejected_probability:.2f}%"
        )

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Please check the model files and input values."
        )
        st.exception(error)
