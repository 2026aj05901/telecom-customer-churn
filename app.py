import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Telecom Customer Churn Prediction")

MODEL_DIR = Path("model")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "support_vector_machine.joblib",
}

model_name = st.selectbox("Select a trained model", list(model_files.keys()))
uploaded_file = st.file_uploader("Upload test CSV (must include Churn)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "Churn" not in data.columns:
        st.error("The uploaded CSV must contain a 'Churn' column.")
    else:
        y_test = data["Churn"].astype(str)
        X_test = data.drop(columns=["Churn"])

        model_path = MODEL_DIR / model_files[model_name]

        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)

            # Keep the feature columns used when the model was trained.
            expected_columns = list(model.feature_names_in_)
            missing = [col for col in expected_columns if col not in X_test.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X_test = X_test[expected_columns]
                predictions = model.predict(X_test)

                classes = list(model.classes_)
                yes_index = classes.index("Yes")
                probabilities = model.predict_proba(X_test)[:, yes_index]

                y_binary = (y_test == "Yes").astype(int)
                metrics = {
                    "Accuracy": accuracy_score(y_test, predictions),
                    "AUC": roc_auc_score(y_binary, probabilities),
                    "Precision": precision_score(y_test, predictions, pos_label="Yes", zero_division=0),
                    "Recall": recall_score(y_test, predictions, pos_label="Yes", zero_division=0),
                    "F1 Score": f1_score(y_test, predictions, pos_label="Yes", zero_division=0),
                    "MCC": matthews_corrcoef(y_test, predictions),
                }

                st.subheader("Evaluation Metrics")
                st.dataframe(
                    pd.DataFrame([metrics]).round(4),
                    use_container_width=True
                )

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
                fig, ax = plt.subplots()
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No", "Yes"],
                    yticklabels=["No", "Yes"],
                    ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Classification Report")
                st.text(classification_report(
                    y_test, predictions, labels=["No", "Yes"], zero_division=0
                ))