import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

df = pd.read_csv("spam_clean.csv")

X = df["Message"]
y = df["Category"]

# Creating train test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction / vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create Model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model - make predictions on test set
y_preds = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)

print("Accuracy: ", accuracy)
print("F1 score: ", f1)
print("Recall", recall)
print("Precision", precision)