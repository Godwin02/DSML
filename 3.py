import  numpy as np

matrix=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix)
U,S,VT=np.linalg.svd(matrix)
print("\nThe realtionship Matrix",U)
print("\nThe diagonal matrix\n",np.diag(S))
print("\nThe column wise relatiionship matrix",VT)
print("Reconstructed matrix\n",U @ np.diag(S) @ VT)