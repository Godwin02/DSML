from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(max_depth=3)
dt.fit(x_train,y_train)
predict=dt.predict(x_test)
accuracy_score=accuracy_score(y_test,predict)
classification_report=classification_report(y_test,predict)
print(classification_report)
new_data=[[3,4,5,2]]
pre=dt.predict(new_data)
target=data.target_names[pre[0]]
print(target)
fig=plt.figure(figsize=(25,20))
plot_tree(dt,feature_names=data.feature_names,class_names=data.target_names,filled=True)
fig.savefig("dt.png")