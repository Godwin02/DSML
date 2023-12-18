import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)
predict=dt.predict(x_test)
accuracy_score=accuracy_score(y_test,predict)
classification_report=classification_report(y_test,predict)
print(classification_report)
print(accuracy_score)
fig=plt.figure(figsize=(25,20))
plot_tree(dt,filled=True,feature_names=data.feature_names,class_names=data.target_names)
fig.savefig("dt1.png")
new_data=[[4,5,6,7,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6]]
new_prediction=dt.predict(new_data)
target=data.target_names[new_prediction[0]]
print(target)