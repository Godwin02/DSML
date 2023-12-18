from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data=load_digits()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
predict=knn.predict(x_test)
accuracy_score=accuracy_score(y_test,predict)
print(accuracy_score)
new_data=[[4,5,6,7,6,6,6,6,8,6,6,66,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,6,6,6,6,66,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,66,6,6,6,6,6,6,6,6]]
new_p=knn.predict(new_data)
target=data.target_names[new_p[0]]
print(target)