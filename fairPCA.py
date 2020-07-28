#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd 

from sklearn import decomposition
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import sklearn


# In[2]:


from scipy.linalg import sqrtm
from numpy import linalg as LA


# In[3]:


def get_pca(X, n):
    pca = decomposition.PCA(n_components=n)
    pca.fit(X)
    return pca


def rec(x_arr, pca):
    x_t = pca.transform(x_arr)
    x_reconst = pca.inverse_transform(x_t)
    err = np.sum((x_arr - x_reconst)**2, axis=1)
    return np.sum(err)


def reconstruction_error(x_arr, pca):
    return rec(x_arr, pca) / len(x_arr)


def loss(x_group, x_all, n):
    pca_all = get_pca(x_all, n)
    pca_group = get_pca(x_group, n)
    res = reconstruction_error(x_group, pca_all) - reconstruction_error(x_group, pca_group)
    assert res >= 0
    return res

def reco(A, B):
    reVal = LA.norm(A-B, 'fro') ** 2
    return reVal / len(B)


# In[4]:


# substitute LFWprocess with any other function that preprocesses your data
# and outputs three centered matrices M, A, B

get_ipython().run_line_magic('run', 'creditProcess.ipynb')
#%run LFWProcess.ipynb


# In[5]:


featureNum = 21


# In[6]:


recons_A = [reconstruction_error(A, get_pca(M, i)) for i in range(1,20)]
recons_B = [reconstruction_error(B, get_pca(M, i)) for i in range(1,20)]


# In[52]:


a=get_pca(M, 20)
x_t = a.transform(A)
x_reconst = a.inverse_transform(x_t)
x_reconst


# In[7]:


plt.figure(figsize=(6,4))
plt.cla()
plt.plot(recons_A, c='b', label='low-ed') # low education
plt.plot(recons_B, c='r', label='high-ed') #
# plt.plot([reconstruction_error(X_nrm, get_pca(X_nrm, i)) for i in range(1,20)], c='g')
plt.legend()
plt.show()


# In[8]:


loss_A = [loss(A, M, i) for i in range(1,20)]
loss_B = [loss(B, M, i) for i in range(1,20)]


# In[9]:


plt.figure(figsize=(6,4))
plt.cla()
plt.title("Average Loss on Vanilla PCA")
plt.plot(loss_A, c='b', label='low-ed') # low education
plt.plot(loss_B, c='r', label='high-ed') #
plt.legend()
plt.show()


# In[10]:


def optApprox(M, d):
    pca = decomposition.PCA(n_components=d)
    pca.fit(M)
    return pca.transform(M)


# In[33]:


def oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2):
    if A.shape != (m_A, n) or B.shape != (m_B, n):
        raise "Input has wrong size"
    
    covA = np.dot(A.T, A)
    covB = np.dot(B.T, B)
    
    o1 = np.sqrt(w_1/m_A) * A
    o2 = np.sqrt(w_2/m_B) * B

    o = np.concatenate([o1, o2], axis=0)

    coeff_P_o = get_pca(o, n=d).components_.T

    P_o = np.dot(coeff_P_o, coeff_P_o.T)
    z_1 = (1/m_A) * (alpha - np.sum(np.multiply(covA, P_o)))
    z_2 = (1/m_B) * (beta - np.sum(np.multiply(covB, P_o)))
    print("z_1:"+str(z_1))
    print("z_2:"+str(z_2))
    #return P_o, z_1, z_2, coeff_P_o
    return P_o, z_1, z_2


# In[26]:


from IPython.display import display, HTML

def mw(A, B, d, eta, T):
    
    covA = np.dot(A.transpose(), A)
    covB = np.dot(B.transpose(), B)
    
    m_A = A.shape[0]
    m_B = B.shape[0]
    n = A.shape[1]
    
    Ahat = optApprox(A, d)
    alpha = LA.norm(Ahat, 'fro') ** 2
    
    Bhat = optApprox(B, d)
    beta = LA.norm(Bhat, 'fro') ** 2
    
    w_1 = 0.5
    w_2 = 0.5
    
    P = np.zeros([n, n])
    
    
    record = [("iteration", "w_1", "w_2", "loss A", "loss B", "loss A by average", "loss B by average")];
    
    for t in range(1, T + 1):
        #P_temp, z_1, z_2, cP_temp = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)
        P_temp, z_1, z_2 = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)
        
        w_1star = w_1 * np.exp(eta * z_1)
        w_2star = w_2 * np.exp(eta * z_2)
        
        w_1 = w_1star / (w_1star + w_2star)
        w_2 = w_2star / (w_2star + w_1star)
        
        P = P + P_temp
        
        
        P_average = P / t
        
        record.append((t, w_1, w_2, z_1, z_2, (1/m_A)*(alpha - np.sum(np.multiply(covA, P_average))), (1/m_B)*(beta - np.sum(np.multiply(covB, P_average)))))
    
    P = P/T
    
    
    z_1 = 1/(m_A)*(alpha - np.sum(np.multiply(covA, P)))
    z_2 = 1/(m_B)*(beta - np.sum(np.multiply(covB, P)))
    z = max(z_1,z_2);
    
    P_last = P_temp
    
    
    zl_1 = 1/(m_A)*(alpha - np.sum(np.multiply(covA, P_last)))
    zl_2 = 1/(m_B)*(beta - np.sum(np.multiply(covB, P_last)))
    z_last = max(zl_1,zl_2)
    print(z)
    print(z_1)
    print(z_2)
    print("Done")
    print('MW method is finished. The loss for group A is ', z_1, 'For group B is ', z_2)
    print("Record:")
    
    for r in record: 
        print("\t\t\t".join([str(rr) for rr in r]))
    return P_last, z, P, z_last


# In[13]:


def fairPCA(A, B, d):
    
    T = 5
    eta = 20
    feature_num = A.shape[1]
    
    P_fair, z, P_last, z_last = mw(A, B, d, eta, T)
    
    if z < z_last:
        P_smart = P_fair
        
    else:
        P_smart = P_last
        
    
    P_smart = np.eye(P_smart.shape[0]) - sqrtm(np.eye(P_smart.shape[0]) - P_smart)
    
    approxFair_A = A @ P_smart;
    approxFair_B = B @ P_smart;
    
    return approxFair_A, approxFair_B, P_smart


# In[34]:


approxFair_A, approxFair_B, P_smart = fairPCA(A, B, featureNum)


# In[36]:


approxFair_A


# In[ ]:





# In[53]:


x_reconst


# In[47]:


reconsFair_A = np.empty(shape=[0])
reconsFair_B = np.empty(shape=[0])


# In[54]:


#reconsFair_A(ell) = re(approxFair_A, A)/size(A, 1);
#reconsFair_B(ell) = re(approxFair_B, B)/size(B, 1);

#function [reVal] = re(Y,Z)
#% Calculate the reconstruction error of matrix Y with respect to matrix Z
#% Matrix Y and Z are of the same size
#reVal = norm(Y-Z, 'fro')^2;
#end

#reVal = LA.norm(A-B, 'fro') ** 2


reVal_A = LA.norm(approxFair_A-A, 'fro') ** 2
reVal_B = LA.norm(approxFair_B-B, 'fro') ** 2
reconsFair_A = reVal_A/np.size(A, 0)
reconsFair_B = reVal_B/np.size(B, 0)


# In[55]:


reconsFair_A


# In[56]:


recons_A


# In[16]:


reconsFair_A = [reco(approxFair_A[i,:], A[i,:]) for i in range(1,20)]
reconsFair_B = [reco(approxFair_B, B) for i in range(1,20)]
#recons_B = [reconstruction_error(B, get_pca(M, i)) for i in range(1,20)]


# In[49]:


P_smart


# In[ ]:


plt.figure(figsize=(6,4))
plt.cla()
plt.plot(recons_A, c='b', label='low-ed') # low education
plt.plot(recons_B, c='r', label='high-ed') #
plt.plot(reconsFair_A, c='g', label='low-ed-fair')
plt.plot(reconsFair_B, c='y', label='high-ed-fair')
# plt.plot([reconstruction_error(X_nrm, get_pca(X_nrm, i)) for i in range(1,20)], c='g')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


approxFair_A.shape


# In[ ]:


recons_A


# In[17]:


P_last


# In[28]:


A


# In[29]:


B


# In[ ]:




