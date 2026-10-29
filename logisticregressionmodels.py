
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
