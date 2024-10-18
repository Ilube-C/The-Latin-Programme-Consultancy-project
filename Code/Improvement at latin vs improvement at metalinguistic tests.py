# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:26:43 2023

@author: Christian
"""
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
   
cdf = cdf[["Anon", "Metalinguistic Test June 22 (score/20)", "Summer Latin Test /30", 
              "Autumn 22 Test Total (score/40)", "Autumn 22 Language (score/30)", 
              "Metalinguistic Test June 23 (score/20)", "Summer 23 Test Total (score/40)", 
              "Summer 23 Language (score/30)"]]

#plot 1: latin 23 - latin 22 vs meta 23 - meta 22

df = pd.DataFrame()
df['Anon'] = cdf['Anon']
df['Metalinguistic Test June 23 (score/20)'] = cdf['Metalinguistic Test June 23 (score/20)']
df['Latin Improvement'] = 100*(cdf['Summer 23 Language (score/30)']/30) - 100*(cdf['Autumn 22 Language (score/30)']/30)
#df['Meta Improvement'] = 100*(cdf['Metalinguistic Test June 23 (score/20)']/20) - 100*(cdf['Metalinguistic Test June 22 (score/20)']/20)

regplot('Latin Improvement', 'Metalinguistic Test June 23 (score/20)', 'Improvement in Latin vs perfromance at Meta', df)

'''
df = df.dropna()

ax = sns.scatterplot(data=df, x="Latin Improvement", 
                 y="Meta Improvement")

ax = sns.regplot(data=df, scatter=False, x="Latin Improvement",
                 y="Meta Improvement", color = "black")

r2 = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[0]
pval = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (-15, 50))
plt.annotate("P-Value = {:.8f}".format(pval), (-15, 45))
plt.title("Change in Latin score vs Change in Metalinguistic test score 22-23")
plt.show()

print(pearsonr(df["Latin Improvement"], df["Meta Improvement"])) 
'''
#plot 2: latin 23 - latin 22 (language) vs meta 23 - meta 22

'''
df = pd.DataFrame()
df['Anon'] = cdf['Anon']
df['Latin Improvement'] = 100*(cdf['Summer 23 Language (score/30)']/30 - cdf['Summer Latin Test /30']/30)
df['Meta Improvement'] = 100*(cdf['Metalinguistic Test June 23 (score/20)']/20 - cdf['Metalinguistic Test June 22 (score/20)']/20)

df = df.dropna()

ax = sns.scatterplot(data=df, x="Latin Improvement", 
                 y="Meta Improvement")

ax = sns.regplot(data=df, scatter=False, x="Latin Improvement", 
                 y="Meta Improvement", color = "black")

r2 = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[0]
pval = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (-8, 50))
plt.annotate("P-Value = {:.8f}".format(pval), (-8, 45))
plt.title("Change in Latin language score vs Change in Metalinguistic test score 22-23")
plt.show()

print(pearsonr(df["Latin Improvement"], df["Meta Improvement"]))

'''

#plot 3: latin summer 23 - latin autumn 22 (same academic year) vs meta summer 23 - meta summer 22
'''
df = pd.DataFrame()
df['Anon'] = cdf['Anon']
df['Latin Improvement'] = 100*((cdf['Summer 23 Language (score/30)'] + cdf['Autumn 22 Language (score/30)'])/60 - cdf['Summer Latin Test /30']/30)
df['Meta Improvement'] = 100*(cdf['Metalinguistic Test June 23 (score/20)']/20 - cdf['Metalinguistic Test June 22 (score/20)']/20)

df = df.dropna()

ax = sns.scatterplot(data=df, x="Latin Improvement", 
                 y="Meta Improvement")

ax = sns.regplot(data=df, scatter=False, x="Latin Improvement", 
                 y="Meta Improvement", color = "black")

r2 = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[0]
pval = pearsonr(df["Latin Improvement"], df["Meta Improvement"])[1]

plt.annotate("R^2-Value = {:.3f}".format(r2), (-15, 50))
plt.annotate("P-Value = {:.8f}".format(pval), (-15, 45))
plt.title("Change in Latin score (language) vs Change in Metalinguistic test score 22-23")
plt.show()

print(pearsonr(df["Latin Improvement"], df["Meta Improvement"])) 


#make plots for:
# metalinguistic test june 23 vs summer 23 test total score
# metalinguistic test june 23 vs summer 23 language score
# add class field to data    

'''