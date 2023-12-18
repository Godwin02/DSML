from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_teat=train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
predict=knn.predict(x_test)
classification_report=classification_report(y_teat,predict)
accuracy_score=accuracy_score(y_teat,predict)
print(classification_report,"\n",accuracy_score)
new_data=[[3,4,5,6]]
predictions=knn.predict(new_data)
target=data.target_names[predictions[0]]
print(target)