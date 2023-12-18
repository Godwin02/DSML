import numpy as np
matrix1=[]
matrix2=[]
def matrix():
    rows=int(input("Enter the number of rows: "))
    coloumns=int(input("Enter the number of columns: "))
    matrix=[]
    for i in range(rows):
        r=[]
        for j in range(coloumns):
            element=int(input(f"Enter the elements at {i} and columns {j}"))
            r.append(element)
        matrix.append(r)
    return np.array(matrix)
print("Matrix1")
matrix1=matrix()
print("Matrix2")
matrix2=matrix()
print(matrix1)
print(matrix2)
add=np.add(matrix1,matrix2)
sub=np.subtract(matrix1,matrix2)
mul=np.multiply(matrix1,matrix2)
div=np.divide(matrix1,matrix2)
print(add,"\n",sub)