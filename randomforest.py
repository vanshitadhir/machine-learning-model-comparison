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
