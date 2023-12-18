from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

x_train=np.array([2,3,4,5,6,7,8,76,5,4,32]).reshape(-1,1)
y_train=np.array([5,6,5,6,6,78,9,11,4,33,3])
l=LinearRegression()
l.fit(x_train,y_train)
predict=l.predict(np.array([5,6,7,8,9,8,3,5]).reshape(-1,1))
print(predict)
slope=l.coef_[0]
intercept=l.intercept_
print("slope",slope)
print("Intercept",intercept)