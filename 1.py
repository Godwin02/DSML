import matplotlib.pyplot as plt

categories=["a",'b','c','d']
values=[4,3,7,5]
plt.bar(categories,values)
plt.xlabel('categories')
plt.ylabel('values')
plt.title("Bar Diagram")
plt.show()