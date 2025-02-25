import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])  # Labels: 0 or 1

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
X_test = np.array([[4, 4]])
y_pred = knn.predict(X_test)

print("Predicted class:", y_pred)

