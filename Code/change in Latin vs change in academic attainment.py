# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:42:29 2023

@author: Christian
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DF3P22 = pd.read_csv('21-22 Argyle Progress Data - Anonymised - 3P.csv')
DF4E22 = pd.read_csv('21-22 Argyle Progress Data - Anonymised - 4E.csv')
DF5A22 = pd.read_csv('21-22 Argyle Progress Data - Anonymised - 5A.csv')
DF5Q22 = pd.read_csv('21-22 Argyle Progress Data - Anonymised - 5Q.csv')
DF3G23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 3G.csv')
DF3P23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 3P.csv')
DF4O23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 4O.csv')
DF5Q23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 5Q.csv')
DF6D23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 6D.csv')
DF6S23 = pd.read_csv('22-23 Argyle Progress Data - Anonymised - 6S.csv')

DF6S23 = DF6S23.drop('Metalinguistic Levels', axis=1)
DF6D23 = DF6D23.drop('Book Notes 23/03/23', axis=1)

dfs = [DF3P22, DF4E22, DF5A22, DF5Q22, DF3G23, DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23]

for i in range(len(dfs)):
    dfs[i].rename(columns={dfs[i].columns[0]: "Anon"}, inplace=True)
    #dfs[i].dropna(subset=['Anon'], inplace=True)
    
dfs22 = [DF3G23, DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23]

for i in range(len(dfs22)):
    dfs22[i].rename(columns={dfs22[i].columns[1]: "Autumn 22 Test Total (score/40)",
                           dfs22[i].columns[2]: "Autumn 22 Language (score/30)",
                           dfs22[i].columns[3]: "Autumn 22 Topic (score/10)",
                           dfs22[i].columns[4]: "Autumn 22 Level",
                           dfs22[i].columns[5]: "Notes",
                           dfs22[i].columns[6]: "Action",
                           dfs22[i].columns[7]: "Metalinguistic Test June 23 (score/20)",
                           dfs22[i].columns[8]: "Metalinguistic Levels",
                           dfs22[i].columns[9]: "Notes",
                           dfs22[i].columns[10]: "Summer 23 Test Total (score/40)",
                           dfs22[i].columns[11]: "Summer 23 Language (score/30)",
                           dfs22[i].columns[12]: "Summer 23 Topic (score/30)",
                           dfs22[i].columns[13]: "Summer 23 Level",
                           dfs22[i].columns[14]: "Notes",
                           dfs22[i].columns[15]: "Action",
                           dfs22[i].columns[16]: "Year Attainment Grade",
                           dfs22[i].columns[17]: "Year Effort Grade",}, inplace=True)

df21 = pd.concat([DF3P22, DF4E22, DF5A22, DF5Q22])
df22 = pd.concat([DF3G23, DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23])

cdf = df21.merge(df22, on='Anon', how = "outer")# left_on="Anonymisation code number", right_on = "Anonymisation code number")

cdf = cdf[["Anon", "Attainment", "Summer Latin Test /30", "Autumn 22 Test Total (score/40)", "Summer 23 Test Total (score/40)", "Year Attainment Grade"]]

key = {"AT":3,
       "GD":2,
       "WT":1}