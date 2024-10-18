# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:26:02 2023

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
                           dfs22[i].columns[12]: "Summer 23 Topic (score/10)",
                           dfs22[i].columns[13]: "Summer 23 Level",
                           dfs22[i].columns[14]: "Notes",
                           dfs22[i].columns[15]: "Action",
                           dfs22[i].columns[16]: "Year Attainment Grade",
                           dfs22[i].columns[17]: "Year Effort Grade",}, inplace=True)

df21 = pd.concat([DF3P22, DF4E22, DF5A22, DF5Q22])
df22 = pd.concat([DF3G23, DF3P23, DF4O23, DF5Q23, DF6D23, DF6S23])

cdf = df21.merge(df22, on='Anon', how = "outer")# left_on="Anonymisation code number", right_on = "Anonymisation code number")

#cdf = df21.merge(df22, on='Anon', how = "inner")# left_on="Anonymisation code number", right_on = "Anonymisation code number")

cdf["Autumn 22 Topic (score/10)"] = pd.to_numeric(cdf["Autumn 22 Topic (score/10)"], errors="coerce")
#cdf["Autumn 22 Topic (score/10)"] = cdf["Autumn 22 Topic (score/10)"].astype(np.float64, errors="coerce")

cdf["Summer 23 Topic (score/10)"] = pd.to_numeric(cdf["Summer 23 Topic (score/10)"], errors="ignore")

cdf["Year Attainment Grade"] = cdf["Year Attainment Grade"].apply(lambda x: x.strip() if type(x) == str else x)
#cdf = cdf[["Anon", "Attainment", "Effort", "Metalinguistic Test June 22 (score/20)", "Summer Latin Test /30"]]

def barchart(categorical_field, numerical_field, title, cdf):
    cdf = cdf[["Anon", categorical_field, numerical_field]]
    cdf = cdf.dropna()

    x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=[categorical_field])[categorical_field])))))
    counts = []
    y = []

    for category in x:
        sum = 0
        count = 0
        for index, row in cdf.iterrows():
            if row[categorical_field] == category:
                sum+=int(row[numerical_field])
                count+=1
        #print(sum, count)
        counts.append(count)
        y.append(sum/count)
    
    plot = plt.bar(x, y)
    plt.ylim(plt.ylim()[0], plt.ylim()[1]+2)
    plt.title(title)
    
    for bar in plot.patches:
        plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
                       (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        
def boxplot(categorical_field, numerical_field, title, cdf):
    cdf = cdf[["Anon", categorical_field, numerical_field]]
    
    cdf = cdf.dropna()
    
    dfg = cdf.groupby(categorical_field)

    counts = [len(v) for k, v in dfg]
    total = float(sum(counts))
    #cases = len(counts)

    widths = [c/total for c in counts]  
    
    fig, ax = plt.subplots(figsize=(12,8))

    cax = cdf.boxplot(column=numerical_field, by=categorical_field, widths=widths, ax=ax)
    cax.set_xticklabels(['%s\n$n$=%d'%(k, len(v)) for k, v in dfg])
    #cax.set_yticklabels([title], fontsize=10)

'''
    x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=[categorical_field])[categorical_field])))))
    #fig, ax = plt.subplots()
    
    data = {}
    for category in x:
        data[category] = []
        for row in cdf.iterrows():
            #print(category)
            #print(row[1][1])
            if row[1][1] == category:
                data[category].append(row[1][2])
        
        
    fig, ax = plt.subplots()
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())    
    plt.title(title)
    
   ''' 
    
   # plt.show()
#======================================================================================================
'''
#plot 1 summer latin grade 22 by attainment

cdf = cdf[["Anon", "Attainment", "Summer Latin Test /30"]]
cdf = cdf.dropna()

x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=["Attainment"])["Attainment"])))))
counts = []
y = []

for category in x:
    sum = 0
    count = 0
    for index, row in cdf.iterrows():
        if row['Attainment'] == category:
            sum+=int(row['Summer Latin Test /30'])
            count+=1
    #print(sum, count)
    counts.append(count)
    y.append(sum/count)

plot = plt.bar(x, y)
plt.title("Latin score by attainment 22")

for bar in plot.patches:
    plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
'''
#==================================================================================================
'''
#plot 2 meta grade 22 by attainment
cdf = cdf[["Anon", "Attainment", "Metalinguistic Test June 22 (score/20)"]]
cdf = cdf.dropna()

x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=["Attainment"])["Attainment"])))))
counts = []
y = []

for category in x:
    sum = 0
    count = 0
    for index, row in cdf.iterrows():
        if row['Attainment'] == category:
            sum+=int(row['Metalinguistic Test June 22 (score/20)'])
            count+=1
    #print(sum, count)
    counts.append(count)
    y.append(sum/count)

plot = plt.bar(x, y)
plt.title("Meta score by attainment 22")

for bar in plot.patches:
    plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

'''
#==========================================================================================================
'''
#plot 3 latin grade 22 by effort
cdf = cdf[["Anon", "Effort", "Summer Latin Test /30"]]
cdf = cdf.dropna()

x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=["Effort"])["Effort"])))))
counts = []
y = []

for category in x:
    sum = 0
    count = 0
    for index, row in cdf.iterrows():
        if row['Effort'] == category:
            sum+=int(row['Summer Latin Test /30'])
            count+=1
    #print(sum, count)
    counts.append(count)
    y.append(sum/count)

plot = plt.bar(x, y)
plt.title("Latin score by Effort 22")

for bar in plot.patches:
    plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

'''
#======================================================================================================
'''
#plot 4 meta grade 22 by Effort
cdf = cdf[["Anon", "Effort", "Metalinguistic Test June 22 (score/20)"]]
cdf = cdf.dropna()

x = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=["Effort"])["Effort"])))))
counts = []
y = []

for category in x:
    sum = 0
    count = 0
    for index, row in cdf.iterrows():
        if row['Effort'] == category:
            sum+=int(row['Metalinguistic Test June 22 (score/20)'])
            count+=1
    #print(sum, count)
    counts.append(count)
    y.append(sum/count)

plot = plt.bar(x, y)
plt.title("Meta score by Effort 22")

for bar in plot.patches:
    plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
'''
#========================================================================================================