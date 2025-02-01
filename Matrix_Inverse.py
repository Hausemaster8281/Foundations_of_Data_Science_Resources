import numpy as np

# create a 3x3 square matrix
matrix1 = np.array([[1, 3, 5], 
             		[7, 9, 2],
                    [4, 6, 8]])

# find inverse of matrix1
result = np.linalg.inv(matrix1)

print(result)
