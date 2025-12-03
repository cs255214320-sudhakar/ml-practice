#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
x = np.array([1,2,4,3,5])
y = np.array([1,3,4,3,7])
print(x.shape)
print(y.shape)
print(x)
print(y)
x = x.reshape(-1,1)
print(x.shape)
print(x)
model = LinearRegression()
model.fit(x, y)
print(model.predict([[8]]))
y_pred = model.predict(x)
plt.scatter(x, y, color='r', marker='o')
plt.plot(x, y_pred, color='b', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print(model.coef_)
print(model.intercept_)
from sklearn.preprocessing import StandardScaler
saler = StandardScaler()
x_stand = saler.fit_transform(x.astype(float))
sgdr = SGDRegressor(penalty='l2', alpha=0.15, max_iter=1000)
sgdr.fit(x_stand, y)
print(sgdr.predict([[10]]))
y_pred2 = sgdr.predict(x_stand)
plt.scatter(x_stand, y, color='g', marker='o')
plt.plot(x_stand, y_pred2, color='b', linewidth=2)
plt.xlabel('x_normalized')
plt.ylabel('y')
plt.show()
print(sgdr.coef_)
print(sgdr.intercept_)


# In[ ]:




