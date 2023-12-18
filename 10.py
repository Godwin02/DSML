import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=fetch_california_housing()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
l=LinearRegression()
l.fit(x_train,y_train)
predict=l.predict(x_test)
r2_score=r2_score(y_test,predict)
mean_squared_error=mean_squared_error(y_test,predict)
print(pd.DataFrame({"ACTUAL":y_test,"PREDICTED":predict}))
print(r2_score)
print(mean_squared_error)