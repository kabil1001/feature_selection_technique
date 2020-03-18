#!/usr/bin/env python
# coding: utf-8

# In[183]:


import pandas as pd
import numpy as np


# In[184]:


"""Rank the features in car data according to the mutual information, Asymmetric dependency coefficient, 
Normalized gain ratio, Symmetrical uncertainty. 
"""
car_data = pd.read_excel("C:/Users/kxj133/Downloads/task_2/Car_data.xlsx")


# In[185]:


from sklearn.preprocessing  import LabelEncoder


# In[186]:


LE = LabelEncoder()


# In[187]:


for column in car_data.columns.difference(['acceptability','doors','persons']):
    car_data[f'{column}'] = LE.fit_transform(car_data[f"{column}"])

car_data['doors'] = car_data['doors'].replace('5more',5)

car_data['persons'] = car_data['persons'].replace('more',5)

acceptability_map = {'unacc': 1,'acc' : 2,'vgood':3,'good' : 4}
car_data['acceptability'] = car_data['acceptability'].map(acceptability_map)


# In[189]:



def mutual_information(X,Y):
    
    import numpy as np

    def single_entropy(c):
        
        c_normalized = c/float(np.sum(c))
        
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        
        h = -sum(c_normalized * np.log(c_normalized))  
        
        return h

    def joint_entropy(X, Y):
        
        probs = []
        
        for c1 in set(X):
            
            for c2 in set(Y):
                
                probs.append(np.mean((X == c1) & (Y == c2)))
        
        probs_list = [-p * np.log2(p) for p in probs]

        return np.nansum(probs_list)
    
    cx = np.histogram(X, bins = 10)[0]
    cy = np.histogram(Y,bins = 10)[0]
    
    hx = single_entropy(cx)
    hy = single_entropy(cy)
    hxy = joint_entropy(cx,cy)

    
    mutual_info = hx + hy - hxy
    
    return mutual_info,hx,hy,hxy


# In[190]:


mutual_info_df = pd.DataFrame(columns = ['mutual_information','independent_entropy','target_entropy','joint_entropy'],
                              index = car_data.columns.difference(['acceptability']))

for column in  car_data.columns.difference(['acceptability']):
    MI_tuple = mutual_information(car_data[column],car_data['acceptability'])
    mutual_info_df.loc[column,'mutual_information'] = MI_tuple[0]
    mutual_info_df.loc[column,'independent_entropy'] = MI_tuple[1]
    mutual_info_df.loc[column,'target_entropy'] = MI_tuple[2]
    mutual_info_df.loc[column,'joint_entropy'] = MI_tuple[3]


# In[191]:


mutual_info_df


# In[192]:


### Assymentric dependency coefficient

mutual_info_df['ADC'] = mutual_info_df['mutual_information'] / mutual_info_df['target_entropy']


# In[193]:


mutual_info_df


# In[194]:


### Normalized  Gain Ratio 

mutual_info_df['NGR'] = mutual_info_df['mutual_information']/ mutual_info_df['independent_entropy']


# In[195]:


mutual_info_df


# In[196]:


### Symmentrical uncertainity

mutual_info_df['SU'] = 2*(mutual_info_df['mutual_information'] / 
                       (mutual_info_df['target_entropy'] + mutual_info_df['independent_entropy']))


# In[197]:


mutual_info_df


# In[198]:


features_rank = mutual_info_df.rank()


# In[199]:


features_rank


# In[ ]:





# In[ ]:




