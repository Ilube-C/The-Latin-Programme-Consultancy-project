# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:54:35 2023

@author: Christian
"""

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

classes = ["3P", "4E", "5A", "5Q"]
dfs = [DF3P22, DF4E22, DF5A22, DF5Q22]# DF3G23, 
      #DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23]

for i in range(len(dfs)):
    dfs[i].rename(columns={dfs[i].columns[0]: "Anon"})
    dfs[i]['Class'] = classes[i]
    dfs[i] = dfs[i].dropna()
    ax = sns.regplot(data=dfs[i], x="Summer Latin Test /30", y="Metalinguistic Test June 22 (score/20)")
    plt.show()
    
df = pd.concat(dfs)
df1 = df[["Class", "Summer Latin Test /30", "Metalinguistic Test June 22 (score/20)"]]
df1 = df1.dropna()
ax = sns.scatterplot(data=df1, x="Summer Latin Test /30", 
                 y="Metalinguistic Test June 22 (score/20)", hue = "Class")

ax = sns.regplot(data=df1, scatter=False, x="Summer Latin Test /30", 
                 y="Metalinguistic Test June 22 (score/20)", color = "black")

r2 = pearsonr(df1["Summer Latin Test /30"], df1["Metalinguistic Test June 22 (score/20)"])[0]
pval = pearsonr(df1["Summer Latin Test /30"], df1["Metalinguistic Test June 22 (score/20)"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (10, 18))
plt.annotate("P-Value = {:.8f}".format(pval), (10, 17))
plt.title("Metalinguistic test score vs Latin test score 21/22")
plt.show()

print(pearsonr(df1["Summer Latin Test /30"], df1["Metalinguistic Test June 22 (score/20)"]))