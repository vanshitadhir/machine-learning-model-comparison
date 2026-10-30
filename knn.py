
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
