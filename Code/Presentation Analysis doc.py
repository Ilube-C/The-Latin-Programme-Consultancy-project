# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:51:57 2023

@author: Christian
"""

#Latin Project Presentation doc
import numpy as np
import pandas as pd
import seaborn as sns 
sns.set_theme()
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy import stats as st

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

_key = {"AT":1,
       "GD":0,
       "WT":-1}
    
key = {"ABOVE":3,
       "AT":2,
       "BELOW":1,
      }

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1edfcc", "#eb93e1", "#8c8989"]) 
#x['Col2'] if pd.isnull(x['Col1']) else x['Col1']
""" 
Templates for defining new fields:
    
cdf['new_field'] = 100*(cdf['existing_field']/x + cdf['existing_field']/y)
                        
cdf['AttainmentChange'] = cdf['Year Attainment Grade'].map(lambda x: None if pd.isnull(x) else ('N/A' if x.strip() == 'N/A' else key[x])) - cdf['Autumn 22 Level'].map(lambda x: None if pd.isnull(x) else ('N/A' if x.strip() == 'N/A' else key[x]))

cdf['LatinImprovement'] = 100*((cdf['Summer 23 Test Total (score/40)'] - cdf['Autumn 22 Test Total (score/40)'])/40)

cdf['Year grade'] = cdf['Year Attainment Grade'].map(lambda x: None if pd.isnull(x) else ('N/A' if x.strip() == 'N/A' else key[x]))
"""
def count(categorical_field, cdf):
    dict_ = {}
    for key in cdf[categorical_field].unique():
        dict_[key] = 0
    for x in list(cdf[categorical_field]):
        dict_[x] += 1
    
    x = list(map(str, list(dict_.keys()))) 
    y = list(map(int, [dict_[x] for x in dict_.keys()]))

    
    plot = plt.bar(x, y)
    ##plt.ylim(plt.ylim()[0], plt.ylim()[1]+2)
    plt.title(categorical_field)
    
    
    
    
def barchart(categorical_field, numerical_field, title, cdf):
    cdf = cdf[["Anon", categorical_field, numerical_field]]
    cdf.reindex(index=cdf.index[::-1])
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
        plt.annotate('n = {}'.format(counts[plot.patches.index(bar)]), 
                       (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        
        
def stackbarchart(categorical_field1, categorical_field2, title, cdf, percentage = False):
    #ax = df.plot.bar(stacked=True)
    cdf = cdf[["Anon", categorical_field1, categorical_field2]]
    cdf.reindex(index=cdf.index[::-1])
    cdf = cdf.dropna()

    x = list(np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=[categorical_field1])[categorical_field1]))))))
    y = list(np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=[categorical_field2])[categorical_field2]))))))
    x.reverse()
    y.reverse()
    print(x)
    
    weight_counts = {category:2*[0] for category in y}
    print(weight_counts)
    n = [0, 0]    
    i=0
    for category in x:
        for cat in y:
            #n=0
            for index, row in cdf.iterrows():
                if row[categorical_field1] == category and row[categorical_field2] == cat:
                    weight_counts[cat][i]+=1
                    n[i]+=1
        i+=1
    
    if percentage == True:
        for i in range(len(x)):
            for category in y:
                weight_counts[category][i] = 100*weight_counts[category][i]/n[i]
                
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(2)
    print(weight_counts)
    for cat, weight_count in weight_counts.items():
        #print(cat, weight_count)
        p = ax.bar(x, weight_count, width, label=cat, bottom=bottom)
       # print(weight_count[0], p.patches[0].get_x())
        print("\n")
        print(bottom)
        print(p.patches[0].get_x() + p.patches[0].get_width() / 2, 
                        bottom + weight_count[0]/2)
        bottom += weight_count
        '''if percentage == True:
         plt.annotate('{}%'.format(weight_count[0]), (p.patches[0].get_x() + p.patches[0].get_width() / 2, 
                        bottom + weight_count[0]/2), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
      plt.annotate(format('{}%'.format(weight_count[1])), (p.patches[1].get_x() + p.patches[1].get_width() / 2, 
                        bottom + weight_count[1]/2), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        else:
            plt.annotate(format('n = {}'.format(weight_count[0])), (p.patches[0].get_x() + p.patches[0].get_width() / 2, 
                        bottom + weight_count[0]/2), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
            plt.annotate(format('n = {}'.format(weight_count[1])), (p.patches[1].get_x() + p.patches[1].get_width() / 2, 
                        bottom + weight_count[1]/2), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')'''
        #bottom += weight_count
    for i in range(len(x)):
        bottom = 0
        for category in y:
            plt.annotate('{:.3f}%'.format(weight_counts[category][i]), (p.patches[i].get_x() + p.patches[i].get_width() / 2, 
                        bottom - 6 + weight_counts[category][i]/2), ha='center', va='center',
                          size=15, xytext=(0, 8), textcoords='offset points')
            bottom += weight_counts[category][i]
    print("DONE")
    ax.set_title(title)
    ax.legend(loc="upper center")
    print("DONE AGAIN")
    plt.show()    
    
    
def boxplot(categorical_field, numerical_field, title, cdf, reg=False):
    cdf = cdf[["Anon", categorical_field, numerical_field]]
    
    cdf = cdf.dropna()
    
    dfg = cdf.groupby(categorical_field)
    
    print(dfg.head())


    counts = [len(v) for k, v in dfg]
    print(counts)
    total = float(sum(counts))
    #cases = len(counts)
    
    

    widths = [c/total for c in counts]  
    
    fig, ax = plt.subplots(figsize=(12,8))
    #cax = sns.boxplot(data = cdf, x = categorical_field, y = numerical_field, order = order)
    
    color = dict(boxes='black', whiskers='black', medians='red', caps='black')
    
    cax = cdf.boxplot(column=numerical_field, by=categorical_field, widths=widths, ax=ax, color=color, 
                      whiskerprops= dict(linestyle='-',linewidth=2.0
, color='black'))
    cax.set_xticklabels(['%s\n$n$=%d'%(k, len(v)) for k, v in dfg])
    #cax.set_xticklabels(["BELOW\nn={}".format(counts[0]), "AT\nn={}".format(counts[1]), "ABOVE\nn={}".format(counts[2])])
    cax.set_xticklabels(["Started in 21/22\nn={}".format(counts[0]), "Started in 22/23\nn={}".format(counts[1])])
    #cax.set_yticklabels([title], fontsize=10)
    ax.grid(False)
    if reg:
        cdf[categorical_field] = cdf[categorical_field].astype('float')
        cdf[numerical_field] = cdf[numerical_field].astype('float')
        #print(cdf[categorical_field], cdf[numerical_field])
        ax=sns.regplot(data=cdf, scatter=False, x=categorical_field, y=numerical_field)

        r2 = pearsonr(cdf[categorical_field], cdf[numerical_field])[0]
        pval = pearsonr(cdf[categorical_field], cdf[numerical_field])[1]
    
        plt.annotate("R^2-Value = {:.3f}".format(r2), (plt.xlim()[0]+0.5, plt.ylim()[1]-2))
        plt.annotate("P-Value = {:.8f}".format(pval), (plt.xlim()[0]+0.5, plt.ylim()[1]-4))
        plt.title(title)
        plt.show()
    
def regplot(numerical_field1, numerical_field2, title, cdf):
    cdf[numerical_field1] = cdf[numerical_field1].astype('float')
    cdf[numerical_field2] = cdf[numerical_field2].astype('float')
    plotdf = cdf[["Anon", numerical_field1, numerical_field2]]
    plotdf = plotdf.dropna()
    
    ax = sns.scatterplot(data=plotdf, x=numerical_field1, 
                     y=numerical_field2)
    
    ax = sns.regplot(data=plotdf, scatter=False, x=numerical_field1, 
                     y=numerical_field2, color = "black")
    
    r2 = pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])[0]
    pval = pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])[1]
    
    plt.annotate("R^2-Value = {:.3f}".format(r2), (plt.xlim()[0]+1, plt.ylim()[1]-10))
    plt.annotate("P-Value = {:.8f}".format(pval), (plt.xlim()[0]+1, plt.ylim()[1]-20))
    plt.title(title)
    plt.show()
    
    print(pearsonr(plotdf[numerical_field1], plotdf[numerical_field2])) 
    
def histogram(numerical_field, title, cdf, percentiles = [25, 50, 75]):
    plotdf = cdf[[numerical_field]]
    plotdf = plotdf.dropna()
    plotdf.sort_values(numerical_field, 0)
    plotdf.hist(column=numerical_field, bins = 3)#len(plotdf.index)//4)
    x = plotdf[[numerical_field]]
    plt.annotate("Median score = {}".format(np.median(x)), (plt.xlim()[0]+1, plt.ylim()[1]-2))
    plt.annotate("Mean score = {:.3f}".format(np.mean(np.mean(x))), (plt.xlim()[0]+1, plt.ylim()[1]-4))
    plt.annotate("Standard Deviation = {:.3f}".format(np.sqrt(np.mean(np.mean(abs(x - x.mean())**2)))), (plt.xlim()[0]+1, plt.ylim()[1]-6))
    plt.annotate("Mode = {}".format(st.mode(x)[0][0][0]), (plt.xlim()[0]+1, plt.ylim()[1]-8))
    vals = np.percentile(x, percentiles, 0)
    for q in vals:
        plt.axvline(q, color='red')

def displot(numerical_fields, title, cdf):
    plotdf = cdf[numerical_fields]
    plotdf = plotdf.dropna()
    #plotdf.sort_values(numerical_field, 0)
    
    sns.displot(plotdf, kind = "kde")
    
    mean = np.mean(np.mean(plotdf[[numerical_fields[0]]]))
    #plt.annotate("{} mean score = {:.3f}".format(numerical_fields[0], mean), (0,0))
    
    plt.xlabel("Score")
    
    plt.title(title)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'] 
    
    for field in numerical_fields:
        mean = np.mean(np.mean(plotdf[[field]]))
        print("{} mean score = {:.3f}".format(field, mean))
        print(numerical_fields.index(field))
        plt.annotate("{:.3f}".format(mean), 
                     (plt.xlim()[0]+10, plt.ylim()[1]-(0.0005*(numerical_fields.index(field)+2))), 
                     color = colors[numerical_fields.index(field)])

    plt.annotate("Mean scores", 
                     (plt.xlim()[0]+10, plt.ylim()[1]-(0.0005*(1))))
    
    for field in numerical_fields:
        n = plotdf[[field]].count()[0]
        print("{} n = {}".format(field, n))
        print(numerical_fields.index(field))
        plt.annotate("{}".format(n), 
                     (plt.xlim()[1]-30, plt.ylim()[1]-(0.0005*(numerical_fields.index(field)+2))), 
                     color = colors[numerical_fields.index(field)])

    plt.annotate("Sample size", 
                     (plt.xlim()[1]-30, plt.ylim()[1]-(0.0005*(1))))
    
    
    
    plt.show()
    #add:
    #quartiles
    #average, mode, median, iqr
    #numpy.percentile(a, q, axis)
    #numpy.median()
    #numpy.mean()
    #std = sqrt(mean(abs(x - x.mean())**2))
    #mean(abs(x - x.mean())**2)
    
def hisdisplot(numerical_field, title, cdf, percentiles = [25, 50, 75], colour = "blue"):
    plotdf = cdf[[numerical_field]]
    plotdf = plotdf.dropna()
    plotdf.sort_values(numerical_field, 0)
    x = plotdf[[numerical_field]]
    
    sns.displot(plotdf, kde = "True", palette=[colour])
    
    plt.annotate("Median score = {}".format(np.median(x)), (plt.xlim()[0]+1, plt.ylim()[1]-2))
    plt.annotate("Mean score = {:.3f}".format(np.mean(np.mean(x))), (plt.xlim()[0]+1, plt.ylim()[1]-4))
    plt.annotate("Standard Deviation = {:.3f}".format(np.sqrt(np.mean(np.mean(abs(x - x.mean())**2)))), (plt.xlim()[0]+1, plt.ylim()[1]-6))
    #plt.annotate("Mode = {}".format(st.mode(x)[0][0][0]), (plt.xlim()[0]+1, plt.ylim()[1]-8))
    vals = np.percentile(x, percentiles, 0)
    for q in vals:
        plt.axvline(q, color='red')
        
    
    plt.xlabel("Score")
    plt.title(title)
    plt.show()

'''
categorical_field = "Attainment"
title = "Distribution of Year Attainment Grades 22"

cdf = cdf[["Anon", categorical_field]]
cdf = cdf.dropna()

x_ = np.unique(np.array(list(map(str.strip, np.array(cdf.dropna(subset=[categorical_field])[categorical_field])))))
x = x_[::-1]
counts = []


for category in x:
    count = 0
    for index, row in cdf.iterrows():
        if row[categorical_field] == category:
            count+=1
    #print(sum, count)
    counts.append(count)
    
plot = plt.bar(x, counts)
plt.ylim(plt.ylim()[0], plt.ylim()[1]+2)
plt.title(title)
    
for bar in plot.patches:
    plt.annotate(format('n = {}'.format(counts[plot.patches.index(bar)])), 
    (bar.get_x() + bar.get_width() / 2, 
    bar.get_height()), ha='center', va='center',
    size=15, xytext=(0, 8),
    textcoords='offset points')'''