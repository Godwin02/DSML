from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
nb=GaussianNB()
nb.fit(x_train,y_train)
pred=nb.predict(x_test)
accuracy_score=accuracy_score(y_test,pred)
print(accuracy_score)
new=[[4,4,4,5]]
newp=nb.predict(new)
target=data.target_names[newp[0]]
print(target)