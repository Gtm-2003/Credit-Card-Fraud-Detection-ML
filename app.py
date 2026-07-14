import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

# ----------------------------
# 1. Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# 2. Load Model & Data
# ----------------------------
@st.cache_resource
def load_model():
    # Replace with your actual model filename
    return pickle.load(open("xgb_model.pkl", "rb"))

@st.cache_data
def load_data():
    # Loading dataset for the "Auto-fill" feature
    return pd.read_csv("sample_transactions.csv")

model = load_model()
df = load_data()

# Define features used during training
feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ----------------------------
# 3. Sidebar Settings
# ----------------------------
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Fraud Detection Threshold", 0.0, 1.0, 0.35, 0.01)

st.sidebar.info("""
**How to use:**
1. Enter an amount.
2. Choose to simulate a 'Safe' or 'Fraud' pattern.
3. Click Analyze to see the risk score.
""")

# ----------------------------
# 4. Main Interface
# ----------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("AI-powered Transaction Risk Analyzer")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

# =====================================================
# 🔹 SINGLE PREDICTION
# =====================================================
with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Details")
        user_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        
        # Beginner Tip: This helps you test how the model reacts to different behaviors
        pattern_type = st.radio("Simulation Mode:", ["Sample Legitimate Transaction", "Sample Fraudulent Transaction"])
        
        if st.button("🚀 Run Risk Analysis"):
            # Step A: Get a base pattern from the real data
            if pattern_type == "Sample Legitimate Transaction":
                sample_row = df[df['Class'] == 0].sample(1)
            else:
                sample_row = df[df['Class'] == 1].sample(1)
            
            # Step B: Prepare the data (Same way as training)
            # 1. Log Transform the user's amount
            scaled_amount = np.log1p(user_amount)
            
            # 2. Extract features and replace the 'Amount' with our scaled user input
            input_values = sample_row[feature_names].values.copy()
            input_values[0][-1] = scaled_amount # Replace the last column (Amount)
            
            # Step C: Prediction
            prob = model.predict_proba(input_values)[0][1]
            
            # --- DISPLAY RESULTS ---
            with col_result:
                st.subheader("Analysis Result")
                
                # Visual Gauge/Metric
                if prob > threshold:
                    st.error(f"🚨 FRAUD DETECTED")
                    st.metric("Risk Probability", f"{prob*100:.2f}%", delta="HIGH RISK", delta_color="inverse")
                else:
                    st.success(f"✅ TRANSACTION SAFE")
                    st.metric("Risk Probability", f"{prob*100:.2f}%", delta="LOW RISK")

                st.info(f"Detection Threshold: {threshold:.2f}")

                # ----------------------------
                # Prediction Summary
                # ----------------------------
                st.markdown("### 📋 Prediction Summary")

                summary = pd.DataFrame({
                    "Field": [
                        "Transaction Amount",
                        "Simulation Mode",
                        "Fraud Probability",
                        "Prediction"
                    ],
                    "Value": [
                        f"${user_amount:.2f}",
                        pattern_type,
                        f"{prob*100:.2f}%",
                        "Fraud" if prob > threshold else "Legitimate"
                    ]
                })

                st.table(summary)

                # Progress bar for visual appeal
                st.progress(float(prob))

                # Explainability Section
                st.markdown("---")
                st.write("**Why this result?**")
                st.write(f"The prediction is based on a sample {pattern_type.lower()} selected from the dataset. " f"PCA-transformed features (v1-v28) are retained for prediction. " f"The calculated fraud probability is{prob:.2%}.")

# =====================================================
# 🔹 BATCH PREDICTION
# =====================================================
with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Error Checking: Ensure all columns exist
        if all(col in data.columns for col in feature_names):
            if st.button("⚡ Process Batch"):
                # Apply same scaling to the whole file
                data_processed = data.copy()
                data_processed['Amount'] = np.log1p(data_processed['Amount'])
                
                # Run prediction
                probs = model.predict_proba(data_processed[feature_names])[:, 1]
                data['Fraud_Probability'] = probs
                data['Prediction'] = (probs > threshold).astype(int)
                
                st.success("Batch Processing Complete!")
                st.dataframe(data.head())
                
                # Download link
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Flagged Transactions", csv, "results.csv", "text/csv")
        else:
            st.error(f"Error: CSV must contain these columns: {', '.join(feature_names)}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with Python, XGBoost, and Streamlit for ML Beginners.")