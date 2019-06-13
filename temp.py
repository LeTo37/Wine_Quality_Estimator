#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:47:05 2019

@author: panther
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('winequality-red.csv', header =0, sep=';')
df2 = pd.read_csv('winequality-white.csv', header =0, sep=';')
# concatenate the two dataframes
data = pd.concat([df1, df2], axis = 0)

feature_cols = data.columns.to_list()

bins = [0, 5, 7, 10]
labels = [0, 1, 2]

data['quality'] = pd.cut(data['quality'], bins=bins, labels=labels)

# data['binned'] = np.searchsorted(bins, data['quality'].values)

# create a copy of the dataframe
df = data.copy()
# transform column into categorical data
del df['quality']

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['quality'] = le.fit_transform(df.quality)


# testing plots
fig, ax = plt.subplots()
#Note showfliers=False is more readable, but requires a recent version iirc
box = data.boxplot(by='quality', ax=ax) 
yval = np.concatenate([line.get_ydata() for line in box['whiskers']])
eps = 1.0
ymin, ymax = yval.min()-eps, yval.max()+eps
ax.set_ylim([ymin,ymax])
plt.show()























