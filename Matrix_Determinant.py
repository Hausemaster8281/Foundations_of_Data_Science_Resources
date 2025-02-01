import numpy as np

# create a matrix
matrix1 = np.array([[1, 2, 3], 
             		[4, 5, 1],
                    [2, 3, 4]])

# find determinant of matrix1
result = np.linalg.det(matrix1)

print(result)
