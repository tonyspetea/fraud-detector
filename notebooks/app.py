
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('model/fraud_model.pkl')
explainer = joblib.load('model/explainer.pkl')
feature_names = joblib.load('model/feature_names.pkl')
scaler = joblib.load('model/scaler.pkl')

st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Real-Time Fraud Detection System")
st.markdown("Enter transaction details to assess fraud risk instantly.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Transaction details")
    amount = st.number_input("Transaction amount ($)", 0.0, 30000.0, 100.0, step=0.01)
    time = st.number_input("Time (seconds since first transaction)", 0, 172792, 50000)

    st.markdown("**PCA features (V1–V14)**")
    st.caption("These represent anonymized transaction characteristics")
    v_vals = {}
    for i in range(1, 15):
        v_vals[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0, step=0.1, key=f"v{i}")

    st.markdown("**PCA features (V15–V28)**")
    for i in range(15, 29):
        v_vals[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0, step=0.1, key=f"v{i}")

    predict_btn = st.button("Analyse transaction", use_container_width=True)

with col2:
    if predict_btn:
        amount_scaled = scaler.transform([[amount]])[0][0]
        time_scaled = (time - 94813) / 47488

        input_dict = {f"V{i}": v_vals[f"V{i}"] for i in range(1, 29)}
        input_dict["Amount_scaled"] = amount_scaled
        input_dict["Time_scaled"] = time_scaled

        input_df = pd.DataFrame([input_dict])[feature_names]
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("Risk assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{prob*100:.1f}%")
        m2.metric("Transaction amount", f"${amount:,.2f}")
        m3.metric("Risk level", "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW")

        if prob > 0.6:
            st.error("HIGH FRAUD RISK — recommend blocking this transaction")
        elif prob > 0.3:
            st.warning("MEDIUM RISK — recommend additional verification")
        else:
            st.success("LOW RISK — transaction appears legitimate")

        st.subheader("Why this decision?")
        shap_values = explainer.shap_values(input_df)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Overall feature importance")
        st.image("outputs/shap_importance.png")
    else:
        st.info("Set transaction details on the left and click Analyse transaction to see the fraud risk assessment.")
        st.subheader("Model performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", "0.9785")
        m2.metric("PR-AUC", "0.7974")
        m3.metric("Fraud recall", "89%")
        m4.metric("Training samples", "454,902")
        st.image("outputs/confusion_matrix.png")
