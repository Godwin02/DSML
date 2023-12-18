import  pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv('Salary_Data.csv')
x=data['YearsExperience'].values.reshape(-1,1)
y=data['Salary'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
l=LinearRegression()
l.fit(x_train,y_train)
predict=l.predict(x_test)
r2_score=r2_score(y_test,predict)
print(r2_score)
plt.scatter(x_test,y_test,edgecolors="black")
plt.plot(x_test,predict,color='blue')
plt.legend()
plt.show()