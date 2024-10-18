# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:41 2024

@author: Christian
"""
'''
read in google sheet

iterate over every anonymous id


(probably use dict)


Update dictionary based on needed scores from each anonymous id

(concern: order of elements in dictionary)

convert back into spreadsheet 




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
    '''


import pandas as pd

df4 = pd.read_csv('Yearly Report Sheet - Year 4.csv')
df5 = pd.read_csv('Yearly Report Sheet - Year 5.csv')
df6 = pd.read_csv('Yearly Report Sheet - Year 6.csv')

#You may have to standardise the columns in each dataframe before concatenating them
#3 extra columns in CDF possibly result of empty column, surname column, change vs %change

cdf = pd.concat([df4, df5, df6])
cdf["ID"] = ""
n=0
for i, row in cdf.iterrows():
            cdf.at[i,'ID'] = n
            n+=1
                
tracking = {}

for index, row in cdf.iterrows():
    print(row["ID"])
    #print(row["Autumn 23 Test Total (score/40)"])
    tracking[row["ID"]] = [row["ID"], row["Autumn 23 Test Total (score/40)"], row["Summer 23 Test Total (score/40)"], row["Change"]]
    
print(tracking)
    
df = pd.DataFrame.from_dict(tracking, orient='index')
df.columns = ["ID", "Autumn 23 Test Total (score/40)", "Summer 23 Test Total (score/40)", "Change"]
df.to_csv('tracking.csv', index=False) 