#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


data_size = 13232


# In[5]:


img_size = 1764


# In[6]:


fileID1= open('C:/Users/m4531/Desktop/semester2/Labor Data Mining/Fair-PCA-master/sex.txt')


# In[7]:


p = np.loadtxt('C:/Users/m4531/Desktop/semester2/Labor Data Mining/Fair-PCA-master/sex.txt', delimiter=',', skiprows=1)
sex = np.array(p)


# In[26]:


for i in range(1,13233):
    p = np.loadtxt('C:/Users/m4531/Desktop/semester2/Labor Data Mining/Fair-PCA-master/data/images/img'+str(i-1)+'.txt')
    p = p.ravel()
    images=np.append(images,[p],axis=0)


# In[29]:


# normalization
images = images/255


# In[31]:


copy=images


# In[37]:


# center the images
mm = np.mean(images,0)
images_centered= images - mm


# In[38]:


images_centered


# In[104]:


images_female = np.empty(shape=[0,img_size])
images_male = np.empty(shape=[0,img_size])


# In[105]:


#writetable(table(sex),'sex.txt','Delimiter','\t')


for j in range(0,13232):
    if sex[j] == 0:        
        images_female = np.append(images_female,[images[j,:]],axis=0)
    else:
        images_male = np.append(images_male,[images[j,:]],axis=0)


# In[109]:


female_mean = np.mean(images_female, 0)
size_female = np.size(images_female)
images_female_centered = images_female - female_mean


male_mean = np.mean(images_male, 0)
size_male = np.size(images_male)
images_male_centered = images_male - male_mean


# In[110]:


M = images_centered
A = images_female_centered
B = images_male_centered


# In[ ]:




