from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=load_iris()
x=data.data
y=data.target

k=KMeans(n_clusters=3,random_state=42)
k.fit(x)
labels=k.labels_
centroids=k.cluster_centers_
plt.scatter(x[:,0],x[:,1],c=labels,cmap='viridis',marker='^',edgecolors='black')
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,marker='*',label='centroids')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Clustering")
plt.savefig("plot.png")
