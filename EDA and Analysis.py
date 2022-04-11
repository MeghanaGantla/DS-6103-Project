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
tracks = pd.read_csv("song_data.csv")
print(tracks.head())
# %%
len(tracks)
# %%
len(tracks["song_popularity"].unique())
# %%
matrix = tracks.corr()
print(matrix)
print("Correlation plot for artists:")
sns.heatmap(tracks.corr(), annot = True)
plt.show()