import numpy as np

# create two matrices
matrix1 = np.array([[1, 3], 
             		[5, 7]])
             
matrix2 = np.array([[2, 6], 
                    [4, 8]])

# calculate the dot product of the two matrices
result = np.dot(matrix1, matrix2)

print("matrix1 x matrix2: \n",result)
