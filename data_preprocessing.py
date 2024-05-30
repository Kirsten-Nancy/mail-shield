import re
import pandas as pd
import numpy as np

def preprocess_data(text):
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]","", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove urls
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"_", "", text)
    return text

df = pd.read_csv("spam.csv")
df["Message"] = df["Message"].map(preprocess_data)
df["Category"] = df["Category"].map({"spam": 1, "ham": 0})

df.to_csv("spam_clean.csv", encoding="utf-8", index=False)



# # Check for missing values after preprocessing
# df = df.dropna()
# missing_values = df.isnull().sum()
# print(missing_values)

# df.to_csv("spam_clean.csv", encoding="utf-8", index=False)

