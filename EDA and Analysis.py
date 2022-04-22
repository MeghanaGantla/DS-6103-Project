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

#%% 
# Checking datatypes 
df.info() 

#%%
# Describing data
df.describe()

# %%
# unique values in each column
df.nunique().sort_values()

# %%
# checking for NA values
df.isnull().sum()

#%%
# Checking for duplicates
dups = df.duplicated(subset = None, keep = 'first')
duplicates = dups[dups == True].shape[0]
print("Number of observations in the data frame: ", len(df))
print("Number of duplicates: ", str(duplicates))
df.drop_duplicates(inplace=True)
print("Number of observations in the dataframe after dropping duplicates: ", len(df))
