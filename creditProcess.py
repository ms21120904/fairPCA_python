#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


data = pd.read_csv('C:/Users/m4531/Desktop/semester2/Labor Data Mining/Fair-PCA-master/data/credit/default_degree.csv',skiprows=1)


# In[6]:


np.size(data,0)


# In[7]:


data


# In[8]:


data = np.array(data)


# In[9]:


data


# In[10]:


data = data[:,1:np.size(data,1)]
# preprocess the credit data. The output of the function is the centered
# data as matrix M. Centered low educated group A and high educated as
# group B. 


# In[11]:


# vector of sensitive attribute.
sensitive = data[:,0]


# In[12]:


# normalizing the sensitive attribute vetor to have 0 for grad school and 
# university level education and positive value for high school, other
normalized = (sensitive-1)*(sensitive-2)


# In[13]:


normalized


# In[14]:


# getting rid of the colum corresponding to the senstive attribute.
data = data[:,1:22]


# In[15]:


data


# In[16]:


n = np.size(data, 1)


# In[17]:


n


# In[18]:


# centering the data and normalizing the variance across each column
#for i in range(0,n): 


mean = np.mean(data,0)


# In[19]:


data = data - mean


# In[20]:


data


# In[21]:


std = np.std(data,axis=0,ddof=1)


# In[22]:


std


# In[23]:


data = data / std


# In[24]:


#for i in range(0,n):
   #std = np.std(data,axis=0,ddof=1)
   #data[:,i] = data[:,i] / std
data


# In[25]:


low = 0


# In[26]:


data_lowEd=np.empty(shape=[0,n])


# In[27]:


data_highEd=np.empty(shape=[0,n])


# In[28]:


# data for low educated populattion
for i in range(0,np.size(data,0)):
    if normalized[i] == 2:
        #data_lowEd=[]
        #data_lowEd.append([data[i,:]])
        #data_lowEd[i,:] = a
        #data_lowEd=np.insert(data[i,:],1,axis=0)
        #np.append(data_lowEd,[a],axis=0)
        #print(data[i,:])
        #np.row_stack((data_lowEd,data[i,:]))
        #data_lowEd
        #np.append(data_lowEd,[data[i,:]],axis=0)
        a=data[i,:]
        data_lowEd=np.append(data_lowEd,[a],axis=0)
    else:
        data_highEd = np.append(data_highEd,[data[i,:]],axis=0)


# In[37]:


data_highEd.shape


# In[36]:


data.shape


# In[30]:


lowEd_copy = data_lowEd
highEd_copy = data_highEd


# In[31]:


mean_lowEd = np.mean(lowEd_copy,0)
mean_highEd = np.mean(highEd_copy,0)


# In[32]:


# centering data for high- and low-educated
lowEd_copy = lowEd_copy - mean_lowEd
highEd_copy = highEd_copy - mean_highEd


# In[33]:


M = data
A = lowEd_copy
B = highEd_copy


# In[34]:


data_lowEd


# In[35]:


data_highEd


# In[ ]:




