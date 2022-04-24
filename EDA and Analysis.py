#%%
import csv
from cProfile import label
from pickle import TRUE
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

#%%
# Reading data
df = pd.read_csv("song_data.csv")
df.head()

# %%
# # Changing column names
df.rename(columns = {'song_popularity':'popularity', 'song_duration_ms':'duration', 'audio_mode':'mode', 'audio_valence':'valence', 'time_signature':'tsign'}, inplace = True)
df.head()
#%% 
# Checking datatypes 
df.info() 

#%%
# Describing data
df.describe().transpose()

# %%
# Removing irrelevant columns
df.drop(['song_name'], axis = 1, inplace = True)
df.drop(["instrumentalness"], axis = 1, inplace = True)
df.head()


# %%
# Sorting by unique values in each column
df.nunique().sort_values()

# %%
# Checking for null values
print(df.isnull().sum(), "\nThere are no null values in any of the columns")

#%%[markdown]
# # Cleaning Data

#%%
# Checking and removing duplicates
dups = df.duplicated(subset = None, keep = 'first')
duplicates = dups[dups == True].shape[0]
print("Number of observations in the data frame: ", len(df))
print("Number of duplicates: ", str(duplicates))
df.drop_duplicates(inplace=True)
print("Number of observations in the dataframe after dropping duplicates: ", len(df))

# %%
# Correlations
sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0) #, annot = True)
plt.title("Correlations between columns")
plt.figure(figsize=[25,25])
plt.show()
# %%
# Pair plots for all columns

sns.pairplot(df)
plt.show()

# %% 
# Seperating Numerical data
nf = df.drop(["key", "mode", "tsign"], axis = 1)
nf.head()

#%% 
# Seperating Categorical data
cf = df[["key", "mode", "tsign"]]
cf.head()

# %%
# Histograms for Numerical data
fig, axis = plt.subplots(5, 2)
fig.set_size_inches(25, 25)
column = nf.columns
for i in range(5):
    for j in range(2):
        sns.histplot(nf[column[2*i+j]], ax=axis[i, j], color = list(np.random.randint([255,255,255])/255))

#%%
# Count plots for categorical data
sns.countplot(x = "mode", data = df, palette="Set2")
plt.show()
sns.countplot(x = "key", data = df, palette="Set2")
plt.show()
sns.countplot(x = "tsign", data = df, palette="Set2")
plt.xlabel("Time Signature")
plt.show()

# %%
