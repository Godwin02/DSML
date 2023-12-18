from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import  seaborn as sns


data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
nb=GaussianNB()
nb.fit(x_train,y_train)
pred =nb.predict(x_test)
accuracy_score=accuracy_score(y_test,pred)
print(accuracy_score)
new=[[4,5,6,7,9,77,6,9,5,4,3,2,2,4,5,6,79,0,876,54,32,2,34567,89,8,76,543,2,66,6]]
pr=nb.predict(new)
tar=data.target_names[pr[0]]
print(tar)
confusion_matrix=confusion_matrix(y_test,pred)
print(confusion_matrix)
figure=plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix,annot=True,cmap='Blues',fmt='g')#format=general
plt.title("Confusion Matrix")
figure.savefig("fig1.png")
