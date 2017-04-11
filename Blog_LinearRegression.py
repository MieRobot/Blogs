
# coding: utf-8

# In[2]:

# Import and read the datset
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C://Users//Koyel//Desktop/MieRobotAdvert.csv")

dataset.head()


# In[3]:

dataset.describe()


# In[4]:

dataset.columns


# In[5]:

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(dataset)


# In[6]:

sns.heatmap(dataset.corr())


# In[7]:

dataset.columns


# In[8]:

X = dataset[['Facebook', 'Twitter', 'Google']]
y = dataset['Hits']


# In[9]:

from sklearn.model_selection import train_test_split


# In[10]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[11]:

from sklearn.linear_model import LinearRegression


# In[12]:

lm = LinearRegression()


# In[13]:

lm.fit(X_train,y_train)


# In[14]:

print(lm.intercept_)


# In[15]:

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Calculated Coefficient'])
coeff_df


# In[17]:

predictions = lm.predict(X_test)


# In[26]:

plt.ylabel("likes predicted")
plt.title("Likes predicated for MieRobot.com blogs",color='r')
plt.scatter(y_test,predictions)


# In[23]:

print (lm.score)


# In[19]:

sns.distplot((y_test-predictions),bins=50);


# In[20]:

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:



