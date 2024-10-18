# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:26:43 2023

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
    dfs[i].dropna(subset=['Anon'], inplace=True)
    
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
                           dfs22[i].columns[12]: "Summer 23 Topic (score/10)",
                           dfs22[i].columns[13]: "Summer 23 Level",
                           dfs22[i].columns[14]: "Notes",
                           dfs22[i].columns[15]: "Action",
                           dfs22[i].columns[16]: "Year Attainment Grade",
                           dfs22[i].columns[17]: "Year Effort Grade",}, inplace=True)

df21 = pd.concat([DF3P22, DF4E22, DF5A22, DF5Q22])
df22 = pd.concat([DF3G23, DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23])

cdf = df21.merge(df22, on='Anon', how = "inner")# left_on="Anonymisation code number", right_on = "Anonymisation code number")

cdf["Autumn 22 Topic (score/10)"] = pd.to_numeric(cdf["Autumn 22 Topic (score/10)"], errors="coerce")
#cdf["Autumn 22 Topic (score/10)"] = cdf["Autumn 22 Topic (score/10)"].astype(np.float64, errors="coerce")

cdf["Summer 23 Topic (score/10)"] = pd.to_numeric(cdf["Summer 23 Topic (score/10)"], errors="ignore")
'''
plotdf = cdf[["Metalinguistic Test June 23 (score/20)", "Summer 23 Language (score/30)"]]
plotdf = plotdf.dropna()

ax = sns.scatterplot(data=plotdf, x="Summer 23 Language (score/30)", 
                 y="Metalinguistic Test June 23 (score/20)")

ax = sns.regplot(data=plotdf, scatter=False, x="Summer 23 Language (score/30)", 
                 y="Metalinguistic Test June 23 (score/20)", color = "black")

r2 = pearsonr(plotdf["Summer 23 Language (score/30)"], plotdf["Metalinguistic Test June 23 (score/20)"])[0]
pval = pearsonr(plotdf["Summer 23 Language (score/30)"], plotdf["Metalinguistic Test June 23 (score/20)"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (17.5, 18))
plt.annotate("P-Value = {:.8f}".format(pval), (17.5, 17))
plt.title("Metalinguistic test score vs Latin test score (language) 22/23")
plt.show()

print(pearsonr(plotdf["Summer 23 Language (score/30)"], plotdf["Metalinguistic Test June 23 (score/20)"])) 
'''
plotdf = cdf[["Metalinguistic Test June 23 (score/20)", "Summer 23 Language (score/30)", "Autumn 22 Language (score/30)"]]
plotdf = plotdf.dropna()

plotdf['Latin Aggregate'] = 50*(plotdf['Summer 23 Language (score/30)']/30 + plotdf['Autumn 22 Language (score/30)']/30)
plotdf['meta'] = 100*(plotdf['Metalinguistic Test June 23 (score/20)']/20)

ax = sns.scatterplot(data=plotdf, x="Latin Aggregate", 
                 y="meta")

ax = sns.regplot(data=plotdf, scatter=False, x="Latin Aggregate", 
                 y="meta", color = "black")

r2 = pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])[0]
pval = pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (50, 90))
plt.annotate("P-Value = {:.8f}".format(pval), (50, 85))
plt.title("Metalinguistic test score vs Latin Aggregate (language) 22-23")
plt.show()

print(pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])) 

def regplot(numerical_field1, numerical_field2, title, cdf):
    plotdf = cdf[["Anon", numerical_field1, numerical_field2]]
    plotdf = plotdf.dropna()
    
    ax = sns.scatterplot(data=plotdf, x=numerical_field1, 
                     y=numerical_field2)
    
    ax = sns.regplot(data=plotdf, scatter=False, x=numerical_field1, 
                     y=numerical_field2, color = "black")
    
    r2 = pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])[0]
    pval = pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])[1]
    
    plt.annotate("R^2-Value = {:.3f}".format(r2), (plt.xlim()[0]+1, plt.ylim()[1]-2))
    plt.annotate("P-Value = {:.8f}".format(pval), (plt.xlim()[0]+1, plt.ylim()[1]-4))
    plt.title(title)
    plt.show()
    
    print(pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])) 
    
'''

    plotdf['Latin Aggregate'] = 50*(plotdf['Summer 23 Language (score/30)']/30 + plotdf['Autumn 22 Language (score/30)']/30)
    plotdf['meta'] = 100*(plotdf['Metalinguistic Test June 23 (score/20)']/20)
    
    ax = sns.scatterplot(data=plotdf, x="Latin Aggregate", 
                     y="meta")
    
    ax = sns.regplot(data=plotdf, scatter=False, x="Latin Aggregate", 
                     y="meta", color = "black")
    
    r2 = pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])[0]
    pval = pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])[1]
    
    plt.annotate("R^2-Value = {:.3f}".format(r2), (50, 90))
    plt.annotate("P-Value = {:.8f}".format(pval), (50, 85))
    plt.title("Metalinguistic test score vs Latin Aggregate (language) 22-23")
    plt.show()
    
    print(pearsonr(plotdf["Latin Aggregate"], plotdf["meta"])) 
    '''
#make plots for:
# metalinguistic test june 23 vs summer 23 test total score
# metalinguistic test june 23 vs summer 23 language score
# add class field to data    