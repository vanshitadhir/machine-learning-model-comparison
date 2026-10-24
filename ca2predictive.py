# Heart Disease Prediction
# Data Science and Machine Learning Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------

df = pd.read_csv("C:/Users/Vanshita/Downloads/heart dataset.csv")

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Find the target column
# Common names include target, HeartDisease, output and diagnosis
target_names = ["target", "heartdisease", "output", "diagnosis", "num"]

target_col = None

for col in df.columns:
    clean_name = col.lower().replace(" ", "").replace("_", "")

    if clean_name in target_names:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "Target column not found. Check your CSV column names "
        "and set target_col to the correct column."
    )

print("\nTarget column:", target_col)

# Remove rows where the target is missing
df = df.dropna(subset=[target_col])

# Remove an ID column if present
for col in df.columns:
    if col.lower().strip() in ["id", "patientid"]:
        df = df.drop(col, axis=1)

# Fill missing values
for col in df.columns:

    if col == target_col:
        continue

    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# Encode categorical columns into numbers
encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col].astype(str))

# Save cleaned dataset
df.to_csv("cleaned_heart_dataset.csv", index=False)

print("\nCleaned dataset saved.")

