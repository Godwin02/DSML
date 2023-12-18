from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data=load_breast_cancer()
x=data.data
k=KMeans(n_clusters=4,random_state=42)
k.fit(x)
labels=k.labels_
centers=k.cluster_centers_
plt.scatter(x[:,3],x[:,6],c=labels,cmap="viridis",marker='^',edgecolors='black')
plt.scatter(centers[:,3],centers[:,6],s=200,c='red',label="Centroids",marker='*')
plt.savefig("plot2.png")