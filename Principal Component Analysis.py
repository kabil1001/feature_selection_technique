#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[50]:


wine_data = pd.read_excel("C:/Users/kxj133/Downloads/task_2/Wine_quality (1).xlsx")


# In[104]:


def PCA(data,num_components):
    
    from numpy import mean
    from numpy import cov
    from numpy.linalg import eig

    # calculate the mean of each column
    M = mean(data.T, axis=1)

    # center columns by subtracting column means
    C = data - M

    # calculate covariance matrix of centered matrix
    V = cov(C.T)

    # eigendecomposition of covariance matrix
    explained_variance, principal_components = eig(V)

    # project data
    P = principal_components.T.dot(C.T)

    return pd.DataFrame(P[0:num_components].T),pd.DataFrame(principal_components[0:num_components].T),explained_variance[0:num_components]


# In[105]:


pca,components,variance = PCA(wine_data,3)


# In[106]:


pca


# In[107]:


components


# In[108]:


variance

