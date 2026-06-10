# credit_risk_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Prediction Dashboard")
st.write(
    """
    This dashboard predicts whether a loan applicant is likely to default.
    It demonstrates data cleaning, classification modeling, model evaluation,
    and risk interpretation.
    """
)


# -----------------------------
# Generate Demo Dataset
# -----------------------------
def generate_demo_data(n=2000, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(18, 70, n)
    income = np.random.normal(65000, 25000, n).clip(15000, 200000)
    loan_amount = np.random.normal(18000, 9000, n).clip(1000, 60000)
    credit_score = np.random.normal(680, 90, n).clip(300, 850)
    employment_years = np.random.randint(0, 30, n)
    debt_to_income = np.random.uniform(0.05, 0.75, n)

    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN"],
        n,
        p=[0.45, 0.4, 0.15]
    )

    purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "education"],
        n,
        p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )

    # Risk score logic
    risk_score = (
        0.035 * loan_amount / 1000
        + 2.8 * debt_to_income
        - 0.006 * credit_score
        - 0.015 * employment_years
        - 0.00001 * income
    )

    risk_score += np.where(home_ownership == "RENT", 0.25, 0)
    risk_score += np.where(purpose == "small_business", 0.35, 0)
    risk_score += np.random.normal(0, 0.5, n)

    probability_default = 1 / (1 + np.exp(-risk_score))
    default = (probability_default > np.random.uniform(0.25, 0.85, n)).astype(int)

    data = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "loan_amount": loan_amount.round(2),
        "credit_score": credit_score.round(0),
        "employment_years": employment_years,
        "debt_to_income": debt_to_income.round(3),
        "home_ownership": home_ownership,
        "loan_purpose": purpose,
        "default": default
    })

    return data


# -----------------------------
# Load Data
# -----------------------------
st.sidebar.header("1. Data Source")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset uploaded.")
else:
    df = generate_demo_data()
    st.sidebar.info("Using demo credit risk dataset.")


st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write("Dataset shape:", df.shape)


# -----------------------------
# Select Target Column
# -----------------------------
st.sidebar.header("2. Target Column")

possible_targets = df.columns.tolist()

default_target_index = possible_targets.index("default") if "default" in possible_targets else len(possible_targets) - 1

target_col = st.sidebar.selectbox(
    "Select the target column",
    possible_targets,
    index=default_target_index
)

if df[target_col].nunique() != 2:
    st.error("The target column must be binary, for example 0/1 or Yes/No.")
    st.stop()


# Convert target to 0/1 if necessary
y_raw = df[target_col]

if y_raw.dtype == "object":
    y = y_raw.astype("category").cat.codes
else:
    y = y_raw

X = df.drop(columns=[target_col])


# -----------------------------
# Identify Feature Types
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.subheader("Feature Summary")

col1, col2 = st.columns(2)

with col1:
    st.write("Numeric Features")
    st.write(numeric_features)

with col2:
    st.write("Categorical Features")
    st.write(categorical_features)
# -----------------------------
# Model Settings
# -----------------------------
st.sidebar.header("3. Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Logistic Regression", "Random Forest"]
)

test_size = st.sidebar.slider(
    "Test set size",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=9999,
    value=42
)


# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -----------------------------
# Choose Model
# -----------------------------
if model_choice == "Logistic Regression":
    classifier = LogisticRegression(max_iter=1000)
else:
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced"
    )


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])


# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

if hasattr(model.named_steps["classifier"], "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    y_proba = y_pred


# -----------------------------
# Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

st.subheader("Model Performance")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", f"{accuracy:.3f}")
m2.metric("Precision", f"{precision:.3f}")
m3.metric("Recall", f"{recall:.3f}")
m4.metric("F1 Score", f"{f1:.3f}")
m5.metric("ROC-AUC", f"{roc_auc:.3f}")


# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
ax_cm.imshow(cm)
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)


# -----------------------------
# ROC Curve
# -----------------------------
st.subheader("ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()

st.pyplot(fig_roc)


# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance / Model Explanation")

try:
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_classifier = model.named_steps["classifier"]

    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_features = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(encoded_cat_features)

    if model_choice == "Logistic Regression":
        importance_values = fitted_classifier.coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        })
        importance_df["absolute_importance"] = importance_df["importance"].abs()
        importance_df = importance_df.sort_values(
            by="absolute_importance",
            ascending=False
        ).head(15)

    else:
        importance_values = fitted_classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        })
        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        ).head(15)

    st.dataframe(importance_df)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1]
    )
    ax_imp.set_title("Top Feature Importances")
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp)

except Exception as e:
    st.warning(f"Feature importance could not be displayed: {e}")


# -----------------------------
# Applicant Risk Prediction
# -----------------------------
st.subheader("Predict Risk for a New Applicant")

st.write(
    """
    Fill in the applicant information below.
    The model will estimate the probability of default.
    """
)

input_data = {}

for feature in numeric_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())

    input_data[feature] = st.number_input(
        feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

for feature in categorical_features:
    options = df[feature].dropna().unique().tolist()
    input_data[feature] = st.selectbox(feature, options)

new_applicant = pd.DataFrame([input_data])

if st.button("Predict Default Risk"):
    probability = model.predict_proba(new_applicant)[0][1]
    prediction = model.predict(new_applicant)[0]

    st.write("### Prediction Result")

    st.metric("Estimated Probability of Default", f"{probability:.2%}")

    if probability < 0.25:
        risk_level = "Low Risk"
        recommendation = "Approve"
    elif probability < 0.50:
        risk_level = "Medium Risk"
        recommendation = "Manual Review"
    else:
        risk_level = "High Risk"
        recommendation = "Reject or Require Additional Review"

    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Recommendation:** {recommendation}")

    st.dataframe(new_applicant)


# -----------------------------
# Business Interpretation
# -----------------------------
st.subheader("Business Interpretation")

st.write(
    """
    A credit risk model should not only predict default but also support lending decisions.

    Important evaluation logic:

    - **Precision** tells us: among applicants predicted as risky, how many actually defaulted.
    - **Recall** tells us: among applicants who actually defaulted, how many we successfully caught.
    - **ROC-AUC** tells us how well the model separates risky and non-risky applicants overall.

    In real lending, recall is often very important because missing a high-risk borrower
    can cause financial loss. However, precision also matters because rejecting too many
    good applicants can reduce revenue.
    """
)


# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.write("Built with Python, pandas, scikit-learn, matplotlib, and Streamlit.")

#python3 -m streamlit run main.py
