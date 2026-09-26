import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

TARGET = "heart_disease"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "smoking", "diabetes", "bmi"
]

@st.cache_data
def load_default_data():
    return pd.read_csv("heart_disease_dataset.csv")

@st.cache_resource
def train_models(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    # The same preprocessing is used for every model:
    # median imputation -> standardization -> classifier.
    models = {
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5, weights="uniform", p=1
            ))
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=0.1, kernel="rbf", class_weight="balanced",
                probability=True, random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=2000, random_state=42
            ))
        ])
    }

    for model in models.values():
        model.fit(X, y)

    return models

st.title("❤️ Heart Disease Risk Prediction")
st.caption("KNN and SVM classification with feature scaling and recall-focused model selection.")

st.warning(
    "Educational screening tool only — this model is not a medical diagnosis. "
    "A prediction should not replace evaluation by a qualified healthcare professional."
)

with st.sidebar:
    st.header("Model settings")
    uploaded = st.file_uploader(
        "Optional: upload a CSV with the same columns",
        type=["csv"]
    )
    model_name = st.selectbox(
        "Classifier",
        ["SVM", "KNN", "Logistic Regression"],
        index=0
    )
    threshold = st.slider(
        "Positive-risk probability threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lowering the threshold generally increases sensitivity/recall but may increase false positives."
    )

try:
    data = pd.read_csv(uploaded) if uploaded is not None else load_default_data()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

missing_cols = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing_cols:
    st.error("Missing required columns: " + ", ".join(missing_cols))
    st.stop()

models = train_models(data)

with st.expander("Dataset overview"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Positive cases", f"{int(data[TARGET].sum()):,}")
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("Enter clinical attributes")

# Human-friendly explanations for the coded columns.
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=53)
        sex = st.selectbox("Sex (0/1)", [0, 1], index=1)
        cp = st.selectbox("Chest-pain type (cp)", [1, 2, 3, 4], index=1)
        trestbps = st.number_input("Resting blood pressure (trestbps)", 50, 250, 130)
        chol = st.number_input("Cholesterol (chol)", 50, 700, 240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])

    with c2:
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=1)
        thalach = st.number_input("Maximum heart rate (thalach)", 40, 250, 150)
        exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [1, 2, 3], index=1)
        ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])

    with c3:
        thal = st.selectbox("Thalassemia code (thal)", sorted(pd.to_numeric(data["thal"], errors="coerce").dropna().astype(int).unique().tolist()))
        smoking = st.selectbox("Smoking (0/1)", [0, 1])
        diabetes = st.selectbox("Diabetes (0/1)", [0, 1])
        bmi = st.number_input("BMI", 10.0, 60.0, 27.5, step=0.1)

    submitted = st.form_submit_button("Predict risk")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "smoking": smoking,
        "diabetes": diabetes,
        "bmi": bmi
    }])[FEATURES]

    model = models[model_name]
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= threshold)

    st.subheader("Prediction")

    if prediction == 1:
        st.error(f"Higher predicted risk class (probability: {probability:.1%})")
    else:
        st.success(f"Lower predicted risk class (probability: {probability:.1%})")

    st.progress(min(max(probability, 0.0), 1.0))
    st.info(
        f"Classifier: {model_name} | Threshold: {threshold:.2f}. "
        "This output is a model classification, not a diagnosis."
    )

st.divider()
st.markdown(
    "**Feature scaling:** StandardScaler is applied inside the pipeline because "
    "KNN is distance-based and SVM is margin/kernel-based. "
    "The training notebook tunes KNN and SVM using recall as the primary CV metric."
)
