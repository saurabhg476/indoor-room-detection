
# coding: utf-8

# In[145]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

#ignore pandas chained assignment warnings
pd.options.mode.chained_assignment = None


df1 = pd.read_csv("/home/saurabh/machine_learning/wifi_project/raw_data/data_saurabh.csv")
df2 = pd.read_csv("/home/saurabh/machine_learning/wifi_project/raw_data/data_yash.csv")
df3 = pd.read_csv("/home/saurabh/machine_learning/wifi_project/raw_data/data_sayan.csv")


# In[146]:


#list_of_columns = df.describe().transpose()[df.describe().transpose()['count'] > 800.0]['count'].index.values.tolist()
#list_of_columns.remove('207_1_Guest')
#list_of_columns
#df = df[list_of_columns]
list_of_columns = ['207_1','207_3','Home-Airtel-2','Home-Airtel-2_5G','Home-Airtel']
df1 = df1[list_of_columns]
df2 = df2[list_of_columns]
df3 = df3[list_of_columns]

df1.fillna(method='ffill',inplace=True)
df1.fillna(method='bfill',inplace=True)
df2.fillna(method='ffill',inplace=True)
df2.fillna(method='bfill',inplace=True)
df3.fillna(method='ffill',inplace=True)
df3.fillna(method='bfill',inplace=True)

df1['output'] = pd.Series(np.ones(df1.shape[0]), index=df1.index)
df2['output'] = pd.Series(np.ones(df2.shape[0]), index=df2.index) * 2
df3['output'] = pd.Series(np.ones(df3.shape[0]), index=df3.index) * 3

frames = [df1,df2,df3]
df = pd.concat(frames,ignore_index=True)
X = df.iloc[:,0:5]
y = df.iloc[:,5]


#df.columns.values


# In[147]:


# #data visualizations!!!!!
# fig=plt.figure() 
# ax = fig.add_subplot(1,1,1)


# # ax.hist(df['Home-Airtel-2_5G'],bins = 20) 
# # plt.title('histogram of wifi 207_1')
# # plt.xlabel('strenth')
# # plt.ylabel('frequency')
# # plt.show()

# ax.plot(df[df['output']==1]['207_1'],df[df['output']==1]['207_3'],'ro')
# ax.plot(df[df['output']==2]['207_1'],df[df['output']==2]['207_3'],'gx')
# plt.title('wifi strength distribution of 207_1 and 207_3')
# plt.xlabel('207_1')
# plt.ylabel('207_3')
# plt.show()


# In[148]:



#print(X)
#print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.40)
#print(X_train.head(5))
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
print("train-set accuracy: ",model.score(X_train,y_train))
print("test-set accuracy: ",model.score(X_test,y_test))


# In[149]:


#plotting learning curves
#train_sizes, train_scores, valid_scores = \
#learning_curve(linear_model.LogisticRegression() \
#               , X_train, y_train, train_sizes=[50, 80, 110], cv=5)
# print(train_sizes)
# print(train_scores)
# print(valid_scores)

#plt.plot(train_scores, 'ro',valid_scores, 'bs')
#plt.show()

