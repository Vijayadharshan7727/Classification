# ==========================================
# Breast Cancer Prediction App
# Random Forest Classification - Streamlit
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="🎗 AI Cancer Predictor",
    page_icon="🎗",
    layout="wide"
)

# ---------------- HEADER ---------------- #
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🎗 Breast Cancer Prediction App</p>', unsafe_allow_html=True)
st.markdown("### Powered by Random Forest Machine Learning Model")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data

df, dataset = load_data()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.header("🧬 Enter Medical Parameters")

mean_radius = st.sidebar.slider("Mean Radius", 5.0, 30.0, 14.0)
mean_texture = st.sidebar.slider("Mean Texture", 5.0, 40.0, 20.0)
mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 200.0, 90.0)
mean_area = st.sidebar.slider("Mean Area", 200.0, 2500.0, 600.0)
mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.20, 0.10)

# Fill remaining features with dataset mean
input_features = np.array(dataset.data.mean()).reshape(1, -1)
input_features[0][0] = mean_radius
input_features[0][1] = mean_texture
input_features[0][2] = mean_perimeter
input_features[0][3] = mean_area
input_features[0][4] = mean_smoothness

prediction = model.predict(input_features)
probability = model.predict_proba(input_features)

# ---------------- MAIN LAYOUT ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Performance")
    st.metric("Accuracy", f"{accuracy:.2f}")

    st.subheader("🌳 Feature Importance")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(X.columns[:10], importances[:10])
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

with col2:
    st.subheader("🔎 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Prediction: Benign (No Cancer)")
    else:
        st.error("⚠ Prediction: Malignant (Cancer Detected)")

    st.write("### 📈 Prediction Probability")
    st.write(f"Benign: {probability[0][1]*100:.2f}%")
    st.write(f"Malignant: {probability[0][0]*100:.2f}%")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Random Forest | AI Medical App")