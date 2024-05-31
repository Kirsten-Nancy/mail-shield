from pathlib import Path
import streamlit as st
import joblib

st.title("Spam Email Detection")
st.write("Please enter an email to check whether it's spam or not.")

email_text = st.text_area("Email")

def load_model_and_vectorizer(model_name):
    models_dir = Path("models")
    
    model_path = models_dir / model_name
    vectorizer_path = models_dir / "vectorizer.pkl"
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

def run_model(email, model, vectorizer):
    email_transformed = vectorizer.transform([email])
    model_pred = model.predict(email_transformed)

    if model_pred == 1:
        return "This email is spam"
    elif model_pred == 0:
        return "Not a spam email"


model_paths = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Support Vector Machine": "svm_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

selected_model = st.selectbox(
    "**Which model would you like to run?**",
    options=(model_paths.keys()),
    index=None,
    placeholder="Select model"
)

if st.button(":green[Run model]"):
    model, vectorizer = load_model_and_vectorizer(model_paths[selected_model])
    prediction = run_model(email_text, model, vectorizer)
    st.write("Prediction: ", prediction)