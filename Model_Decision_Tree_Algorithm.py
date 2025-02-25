import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Sample dataset
X = np.array([[1, 0], [1, 1], [0, 0], [0, 1], [1, 0]])
y = np.array([0, 1, 0, 1, 0])

# Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# Visualize the decision tree
tree.plot_tree(clf, feature_names=["Feature1", "Feature2"], class_names=["0", "1"], filled=True)

# Predict a sample
X_test = np.array([[1, 1]])
y_pred = clf.predict(X_test)
print("Prediction for [1,1]:", y_pred)

