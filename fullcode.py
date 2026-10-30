
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

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


# ==================================================
# PREPARE DATA FOR MACHINE LEARNING
# ==================================================

# Features and target

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==================================================
# TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# STANDARD SCALER
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# MODEL 1: LOGISTIC REGRESSION
# ==================================================

model1 = LogisticRegression(max_iter=1000)

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("\nLOGISTIC REGRESSION")

print("Accuracy:", accuracy_score(y_test, y_pred1))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))

print("Classification Report:")
print(classification_report(y_test, y_pred1))


# ==================================================
# MODEL 2: K-NEAREST NEIGHBORS (KNN)
# ==================================================

model2 = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print("\nK-NEAREST NEIGHBORS")

print("Accuracy:", accuracy_score(y_test, y_pred2))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))

print("Classification Report:")
print(classification_report(y_test, y_pred2))


# ==================================================
# MODEL 3: NAIVE BAYES
# ==================================================

model3 = GaussianNB()

model3.fit(X_train, y_train)

y_pred3 = model3.predict(X_test)

print("\nNAIVE BAYES")

print("Accuracy:", accuracy_score(y_test, y_pred3))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred3))

print("Classification Report:")
print(classification_report(y_test, y_pred3))


# ==================================================
# MODEL 4: RANDOM FOREST
# ==================================================

model4 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model4.fit(X_train, y_train)

y_pred4 = model4.predict(X_test)

print("\nRANDOM FOREST")

print("Accuracy:", accuracy_score(y_test, y_pred4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred4))

print("Classification Report:")
print(classification_report(y_test, y_pred4))

# ==================================================
# RANDOM FOREST VISUALIZATION
# ==================================================

# Visualize one decision tree from Random Forest

plt.figure(figsize=(20, 10))

plot_tree(
    model4.estimators_[0],
    feature_names=X.columns,
    class_names=["No Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Random Forest - Decision Tree Visualization")
plt.show()


# ==================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model4.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
# ==================================================
# COMPARE ALL FOUR MODELS
# ==================================================

results = {
    "Logistic Regression": accuracy_score(y_test, y_pred1),
    "KNN": accuracy_score(y_test, y_pred2),
    "Naive Bayes": accuracy_score(y_test, y_pred3),
    "Random Forest": accuracy_score(y_test, y_pred4)
}

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

print("\nMODEL ACCURACY COMPARISON:")
print(results_df)

# Find the model with the highest accuracy

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Model:")
print(best_model["Model"])

print(
    "Best Accuracy:",
    round(best_model["Accuracy"] * 100, 2),
    "%"
)

# Plot model comparison

results_df.plot(
    x="Model",
    y="Accuracy",
    kind="bar",
    legend=False
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
 
