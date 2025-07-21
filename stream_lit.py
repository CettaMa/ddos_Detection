# app.py
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Map for class labels
class_label_map = {
    0: "BENIGN",
    1: "Dos Hulk",
    2: "Dos GoldenEye",
    3: "Dos Slowloris",
    4: "Dos Slowhttptest"
}

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open("xgboost_with_scaler.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]

model, scaler = load_model_and_scaler()

# List of required features (in correct order)
selected_features = [' Idle Max', ' Fwd IAT Max', ' Flow IAT Max', ' Fwd IAT Std', 'Idle Mean', ' Idle Min',
                    ' Bwd Packet Length Mean', ' Avg Bwd Segment Size', ' Flow IAT Std', ' Bwd Packet Length Std',
                    'Bwd Packet Length Max', ' Packet Length Std', ' Max Packet Length', ' Packet Length Mean',
                    ' Average Packet Size', ' Packet Length Variance', 'Fwd IAT Total', ' Flow Duration',
                    ' Fwd IAT Mean', ' Bwd Packet Length Min', ' Flow IAT Mean', ' Min Packet Length',
                    ' ACK Flag Count', 'FIN Flag Count', ' Bwd IAT Std', ' Bwd IAT Max', ' Destination Port',
                    ' Down/Up Ratio', ' URG Flag Count', ' Bwd IAT Mean']

st.title("🚦 XGBoost Classification with Scaled Features")
st.write("Upload a CSV file containing the required 30 features.")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Validate required features
    if not all(col in df.columns for col in selected_features):
        missing = [col for col in selected_features if col not in df.columns]
        st.error(f"Missing required columns: {missing}")
    else:
        if st.button("🔍 Predict"):
            try:
                X_input = df[selected_features]
                X_scaled = scaler.transform(X_input)
                preds = model.predict(X_scaled)

                # Map numeric predictions to class names
                preds_named = [class_label_map.get(p, f"Unknown ({p})") for p in preds]

                result_df = df.copy()
                result_df["Prediction"] = preds_named

                st.subheader("✅ Prediction Results")
                st.dataframe(result_df)

                # Plot distribution using Plotly
                class_counts = pd.Series(preds_named).value_counts().reset_index()
                class_counts.columns = ["Class", "Count"]

                # Use custom color mapping (optional)
                class_colors = {
                    "BENIGN": "#2ecc71",
                    "Dos Hulk": "#e74c3c",
                    "Dos GoldenEye": "#f39c12",
                    "Dos Slowloris": "#9b59b6",
                    "Dos Slowhttptest": "#3498db"
                }

                fig = px.bar(
                    class_counts,
                    x="Class",
                    y="Count",
                    text="Count",
                    title="📊 Predicted Class Distribution",
                    color="Class",
                    color_discrete_map=class_colors
                )
                fig.update_layout(xaxis_title="Predicted Class", yaxis_title="Number of Instances")
                st.plotly_chart(fig)

                # Download button
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")



