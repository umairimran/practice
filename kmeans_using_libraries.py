#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
df=pd.read_csv("data.csv")
print(df.head())


# In[17]:





# In[24]:



label_encoder=preprocessing.LabelEncoder()
df["Genre"]=label_encoder.fit_transform(df["Genre"])


# In[40]:


from sklearn.cluster import KMeans
data=list(zip(df["Annual Income (k$)"],df["Spending Score (1-100)"]))


# In[48]:


inter=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    inter.append(kmeans.inertia_)
    
plt.plot(range(1,11),inter,marker='o')
plt.title("Elbow method")
plt.xlabel("NUmber of clusters")
plt.ylabel("Inertia")
plt.show()


kmeans = KMeans(n_clusters=5)
kmeans.fit(data)
plt.scatter(df["Annual Income (k$)"],df["Spending Score (1-100)"], c=kmeans.labels_)
plt.show()


# In[ ]:




