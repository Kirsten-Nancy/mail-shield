from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import joblib

path = Path(__file__).parent / "../data/spam_clean.csv"
df = pd.read_csv(path)

X = df["Message"]
y = df["Category"]

# Creating train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction / vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

def train_and_save_model(model, model_name):
        # Create Model
    model = model()

    # Train the model
    model.fit(X_train, y_train)

    # Test the model - make predictions on test set
    y_preds = model.predict(X_test)

    # Save the model and the vectorizer
    # joblib.dump(vectorizer, 'vectorizer.pkl')
    # joblib.dump(model, f"{model_name}_model.pkl")

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
 
    print(f"Evaluation metrics of {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 score: {f1:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")

train_and_save_model(LogisticRegression, "logistic_regression")
train_and_save_model(SVC, 'svm')
train_and_save_model(RandomForestClassifier, "random_forest")

