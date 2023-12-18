from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt

cat=["alt.atheism","soc.religion.christian","comp.graphics","sci.med"]
twenty_train=fetch_20newsgroups(subset='train',categories=cat,shuffle=True,random_state=42)
vector=TfidfVectorizer()
x=vector.fit_transform(twenty_train.data)
y=twenty_train.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
svm=SVC(kernel='linear',random_state=42)
svm.fit(x_train,y_train)
predict=svm.predict(x_test)
accuracy_score=accuracy_score(y_test,predict)
classification_report=classification_report(y_test,predict)
confusion_matrix=confusion_matrix(y_test,predict)
print("The Accuracy Score: ",accuracy_score)
print("The classification report is \n:",classification_report)
print("The Confusion matrix is: ",confusion_matrix)
fig=plt.figure(figsize=(25,20))
sb.heatmap(confusion_matrix,annot=True,fmt='g')
fig.savefig("cm2.png")
new_data=["jesus christ was a human","he was not a believer"]
new_x=vector.transform(new_data)
new_prediction=svm.predict(new_x)
for i, text in enumerate(new_data):
    new_class=twenty_train.target_names[new_prediction[i]]
    print(new_data[i],new_class)