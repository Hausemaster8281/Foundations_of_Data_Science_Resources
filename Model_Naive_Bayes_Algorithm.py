import numpy as np
from sklearn.naive_bayes import GaussianNB

# Sample dataset
X_train = np.array([[1, 20], [2, 21], [3, 22], [4, 23], [5, 24]])
y_train = np.array([0, 0, 1, 1, 1])

# Train model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict
X_test = np.array([[2, 21]])
y_pred = nb.predict(X_test)

print("Prediction:", y_pred)

