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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
# Reading data
df = pd.read_csv("song_data.csv")
df.head()

# %%
# # Changing column names
df.rename(columns = {'song_duration_ms':'duration', 'audio_mode':'mode', 'audio_valence':'valence', 'time_signature':'tsign'}, inplace = True)
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
#fig, axis = plt.subplots(5, 2)
#fig.set_size_inches(25, 25)
#column = nf.columns
#for i in range(5):
#    for j in range(2):
#        sns.histplot(nf[column[2*i+j]], ax=axis[i, j], color = list(np.random.randint([255,255,255])/255))

#%%
# Histograms for Numerical data
plt.figure(figsize=(25,25))
for i in range(len(nf.columns)):
    plt.subplot(5, 2, i+1)
    sns.histplot(nf[nf.columns[i]], color = list(np.random.randint([255,255,255])/255))
plt.show()

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
# Boxplots for numerical data
plt.figure(figsize=(25,25))
for i in range(len(nf.columns)):
    plt.subplot(5, 2, i+1)
    sns.boxplot(nf.columns[i], data = nf, color = list(np.random.randint([255,255,255])/255))
plt.show()

# we can see quite a few outliers in the data
# %%
# Do regresion analysis 
# FORMULA based
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
#%%
model = ols(formula='popularity ~ key + mode + tsign + duration + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
print( type(model) ) 

modelFit = model.fit()
print( modelFit.summary() )
#%%
#Drop variables which P-value <0.05
model2 = ols(formula='popularity ~ mode + tsign + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
print( type(model2) ) 
model2Fit = model2.fit()
print( model2Fit.summary() )
#%%
#Build linear regression model
df["popularity"].where(df["popularity"] < 99, 99, True)
df['popularity'].value_counts()

x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['popularity']
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.sum(abs(y_pred - y_test)) / len(y_pred))
accu = model.score(X_test, y_test)
print("blah: {:.5f}%".format(accu * 100))


#%%Then bulid tree decision model

# %%
#Error: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
#So need change the part of dataframe
#df['popularity'].value_counts()
x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['popularity']
df["popularity"].where(df["popularity"] < 99, 99, True)
df['popularity'].value_counts()
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
#%%
dtree_popularity1 = DecisionTreeClassifier( random_state=1)
dtree_popularity1.fit(X_train,y_train)
y_test_pred = dtree_popularity1.predict(X_test)
#%%
print(accuracy_score(y_test, y_test_pred))
# print(confusion_matrix(y_test, y_test_pred))
# print(classification_report(y_test, y_test_pred))
# %%
#Then try knn model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit
y_pred = knn.predict(x_popularity )                                                                                                 
knn.score(x_popularity,y_popularity)
# Can't not fit, the reason may be the y have too much values and not dummy(catelogy) variable

# %%
df["popularity"]= [ 1 if i>=66.5 else 0 for i in df.song_popularity ]
df["popularity"].value_counts()
df.info()

#%%
from collections import Counter
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index #filtre
        # store indeces
        outlier_indices.extend(outlier_list_col) #The extend() extends the list by adding all items of a list (passed as an argument) to the end.
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2) 
    
    return multiple_outliers
# %%
df.loc[detect_outliers(df,["popularity","duration","danceability","energy","instrumentalness","liveness","loudness","speechiness","valence"])]
# %%
song_data = df.drop(detect_outliers(df,["popularity","duration","danceability","energy","liveness","loudness","speechiness","valence"]),axis = 0).reset_index(drop = True)

# %%
sns.heatmap(song_data.corr(), vmin=-1, vmax=1, center=0)
plt.show()
# %%
y = df['popularity'].copy()
X = df.drop('popularity', axis=1).copy()
    
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
# %%
# Without outliers

model = LinearRegression()
model.fit(X_train, y_train)

model_acc = model.score(X_test, y_test)

print("Test Accuracy (No Outliers): {:.5f}%".format(model_acc * 100))
# %%
