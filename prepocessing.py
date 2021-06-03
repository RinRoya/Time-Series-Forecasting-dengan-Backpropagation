#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


class MinMaxScaler():
    def __init__(self,min=0,max=1):
        self.min = min
        self.max = max
    def fit_transform(self, df):
        df = df.apply(lambda x: (self.max-self.min)*(x-min(df))/(max(df)-min(df))) + self.min
        return df
    def inverse_transform(self, arr):
        arr = (max(arr)-min(arr))*(arr-self.min)/(self.max-self.min) + min(arr)
        return arr


# In[3]:


def train_test_split_data(df, year, step=7):
    df_train = df[df.index.year!=year]
    df_test = df[df.index.year==year]
    
    data_train = df_train.values    
    df_test = df.iloc[len(df)-len(df_test)-step:]
    data_test = df_test.values
    
    x_train = []
    y_train = []
    for i in range(step, len(data_train)):
        x_train.append(data_train[i-step:i])
        y_train.append(data_train[i])
        
    x_test = []
    y_test = []
    for i in range(step, len(data_test)):
        x_test.append(data_test[i-step:i])
        y_test.append(data_test[i])
        
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

