
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("C:/Users/Vanshita/Downloads/heart dataset.csv")

print("First five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nDataset description:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==================================================
# OBJECTIVE 1: ANALYZE HEART DISEASE DISTRIBUTION
# ==================================================

print("\nHeart disease distribution:")
print(df["target"].value_counts())

df["target"].value_counts().sort_index().plot(kind="bar")

plt.title("Heart Disease Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# ==================================================
# OBJECTIVE 2: ANALYZE AGE AND SEX
# ==================================================

# Age distribution
sns.histplot(df["age"], bins=10, kde=True)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# Age compared with heart disease status
sns.boxplot(x="target", y="age", data=df)

plt.title("Age vs Heart Disease")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# Sex compared with heart disease status
pd.crosstab(
    df["sex"], df["target"]
).plot(kind="bar")

plt.title("Sex vs Heart Disease")
plt.xlabel("Sex (0 or 1)")
plt.ylabel("Number of Patients")
plt.legend(["No Disease", "Heart Disease"])
plt.tight_layout()
plt.show()


# ==================================================
# OBJECTIVE 3: ANALYZE CHEST PAIN AND EXERCISE ANGINA
# ==================================================

pd.crosstab(
    df["chest pain type"], df["target"]
).plot(kind="bar")

plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.legend(["No Disease", "Heart Disease"])
plt.tight_layout()
plt.show()

pd.crosstab(
    df["exercise angina"], df["target"]
).plot(kind="bar")

plt.title("Exercise Angina vs Heart Disease")
plt.xlabel("Exercise Angina (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.legend(["No Disease", "Heart Disease"])
plt.tight_layout()
plt.show()


# ==================================================
# OBJECTIVE 4: ANALYZE MEDICAL MEASUREMENTS
# ==================================================

cols = [
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak"
]

for col in cols:
    sns.boxplot(x="target", y=col, data=df)

    plt.title(f"{col} vs Heart Disease")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()



# ==================================================
# OBJECTIVE 5: CORRELATION ANALYSIS
# ==================================================

corr = df.corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Heart Dataset Correlation Heatmap")
plt.tight_layout()
plt.show()
