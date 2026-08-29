import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Example: predicting pass/fail based on hours studied
hours_studied = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    hours_studied, passed, test_size=0.25, random_state=42
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # gives [P(class 0), P(class 1)]

print("Predictions:", predictions)
print("Probabilities:\n", probabilities)
print("Accuracy:", accuracy_score(y_test, predictions))

# Try a new example: someone who studied 4.2 hours
new_student = np.array([[4.2]])
print("Will they pass?", model.predict(new_student))
print("Probability of passing:", model.predict_proba(new_student)[0][1])

# Inspect the learned coefficients
print("Coefficient (b1):", model.coef_[0][0])
print("Intercept (b0):", model.intercept_[0])