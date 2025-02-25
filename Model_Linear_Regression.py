import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Get model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Prediction
X_new = np.array([[6]])
y_pred = model.predict(X_new)

print("Prediction for X=6:", y_pred)
