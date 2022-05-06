#%%
import csv
from cProfile import label
from pickle import TRUE
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pylab as py
import math
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings 
warnings.filterwarnings('ignore')

#%%
# Reading data
df = pd.read_csv("SpotifyAudioFeaturesNov2018.csv")
df.head()

# %%
# # Changing column names
df.rename(columns = {'duration_ms':'duration', 'time_signature':'tsign', 'popularity':'song_popularity'}, inplace = True)
df.head()

#%% 
# Checking datatypes 
df.info() 

#%%
# Describing data
df.describe().transpose()

# %%
# Sorting by unique values in each column
df.nunique().sort_values()

# %%
# Removing irrelevant columns
df.drop(["track_id"], axis = 1, inplace=True)
df.drop(["artist_name"], axis = 1, inplace=True)
df.drop(["track_name"], axis = 1, inplace=True)
df.head()

# %%
# Correlations (before preprocessing)
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot = True)
#plt.savefig("heatmap.jpg")
plt.title("Correlations between columns")
plt.show()

#%% 
# understanding popularity distribution

# Histogram
plt.figure(figsize=(25,25))
sns.histplot(df["song_popularity"], color="green")
plt.xlabel("Popularity")
plt.title("Histogram for popularity")
#plt.savefig("Popularityhist.jpg")
plt.show()

# Boxplot
plt.figure(figsize=(25,25))
sns.boxplot(df["song_popularity"], color = "yellow")
plt.xlabel("Popularity")
plt.title("Boxplot for popularity")
#plt.savefig("Popularitybox.jpg")
plt.show()

#%%
# Dividing popularity into "Below average - 0" and "Above average - 1" 
print(df['song_popularity'].mean())
df["popularity"]= [1 if i >= 24 else 0 for i in df.song_popularity ]
print(df["popularity"].value_counts())
df.drop(["song_popularity"], axis = 1, inplace=True)

# Count plot for popularity
plt.figure(figsize=(25,12))
sns.countplot(df["popularity"], palette="Set2")
plt.title("Count plot for popularity")
#plt.savefig("Popularitycount.jpg")
plt.show()

# %% 
# Seperating data to numerical and categorical
# Seperating Numerical data
nf = df.drop(["key", "mode", "tsign", "popularity"], axis = 1)
print("Numerical data: \n", nf.head())

# Seperating categorical data
cf = df[["key", "mode", "tsign", 'popularity']]
print("\nCategorical data: \n", cf.head())

#%% Describing all numerical data
nf.describe().transpose()

#%% [markdown]
# # Plots before preprocessing

#%%

# Histograms for Numerical data
print("\nHistograms for numerical data:")
plt.figure(figsize=(25,20))
for i in range(len(nf.columns)):
    plt.subplot(5, 2, i+1)
    sns.histplot(nf[nf.columns[i]], color = list(np.random.randint([255,255,255])/255))
plt.show()

# Boxplots for numerical data
print("Boxplots for numerical data:")
plt.figure(figsize=(25,20))
for i in range(len(nf.columns)):
    plt.subplot(5, 2, i+1)
    sns.boxplot(nf.columns[i], data = nf, color = list(np.random.randint([255,255,255])/255))
plt.show()

# Count plots for categorical data
print("Countplots for categorical data:")
plt.figure(figsize=(25,20))
for i in range(len(cf.columns)):
    plt.subplot(2, 2, i+1)
    sns.countplot(cf[cf.columns[i]], palette="Set2")    
plt.show()

# Pair plots for all columns
plt.figure(figsize = (25,25))
sns.pairplot(df)
plt.show()

#%% [markdown]
# # Preprocessing

#%%
# Checking for null values
print(df.isnull().sum(), "\nThere are no null values in any of the columns")

#%%
# Checking and removing duplicates
dups = df.duplicated(subset = None, keep = 'first')
duplicates = dups[dups == True].shape[0]
print("Number of observations in the data frame: ", len(df))
print("Number of duplicates: ", str(duplicates))
df.drop_duplicates(inplace=True)
print("Number of observations in the dataframe after dropping duplicates: ", len(df))

# %% 
# Removing outliers
df1 = df.copy()
features1 = nf[["acousticness", "danceability", "duration", "energy", "liveness", "loudness", "speechiness", "tempo", "valence" ]]
for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q2 = df1[i].quantile(0.75)
    IQR = Q2 - Q1
    df1 = df1[df1[i] <= (Q2+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
print('Before removal of outliers, The dataset had {} samples.'.format(df.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))
df1.head()

# Seperating Numerical data
nf1 = df1.drop(["key", "mode", "tsign", "popularity"], axis = 1)
print("Numerical data:\n", nf1.head())

# %% [markdown]
# Plots after preprocessing

#%%
# heatmap
print("Correlations:")
plt.figure(figsize=(25,15))
sns.heatmap(df1.corr(), vmin=-1, vmax=1, center=0, annot = True)
plt.title("Correlations between columns")
plt.show()

# Pairplot
print("Pairplots:")
plt.figure(figsize=(25,15))
sns.pairplot(df1)
plt.show()

# Histograms for Numerical data
print("Histograms for numerical data:")
plt.figure(figsize=(25,25))
for i in range(len(nf1.columns)):
    plt.subplot(5, 2, i+1)
    sns.histplot(nf1[nf1.columns[i]], color = list(np.random.randint([255,255,255])/255))
plt.show()

# Boxplots for numerical data
print("Boxplots for numerical data:")
plt.figure(figsize=(25,25))
for i in range(len(nf1.columns)):
    plt.subplot(5, 2, i+1)
    sns.boxplot(nf1.columns[i], data = nf1, color = list(np.random.randint([255,255,255])/255))
plt.show()

# %%
# Splitting data
y = df1['popularity']
X = df1.drop('popularity', axis=1).copy()    
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    
# %%
# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
model_acc = model.score(X_test, y_test)
print("Test Accuracy {:.5f}%".format(model_acc * 100))

#%%
# Logit model 1
model = ols(formula='popularity ~ key + mode + tsign + duration + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence + instrumentalness', data=df)
print(type(model)) 
modelFit = model.fit()
print(modelFit.summary())

#%%
# Logit model 2
#Drop variables which P-value <0.05
model2 = ols(formula='popularity ~ mode + tsign + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
print(type(model2) ) 
model2Fit = model2.fit()
print(model2Fit.summary())

#%%
# Multinomial logit
from statsmodels.formula.api import mnlogit
model3 = mnlogit(formula = "popularity ~ C(mode)+C(tsign)+C(key)+ acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence", data = df1)
model3logitfit = model3.fit()
print(model3logitfit.summary())
# Pseudo R-Squared value - 0.0405

#%%
#Build linear regression model
#df["popularity"].where(df["popularity"] < 99, 99, True)
print(df['popularity'].value_counts())
x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['popularity']
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity, random_state=1)

#%%
# Linear regression model 2
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.sum(abs(y_pred - y_test))/len(y_pred))
print(model.score(X_test, y_test))

#%%
# Decision tree
<<<<<<< HEAD
dtreemodel = DecisionTreeClassifier(random_state=1)
dtreemodel.fit(X_train,y_train)
model_pred_dtree = dtreemodel.predict(X_test)
print("Accuracy: {:.5f}%".format(accuracy_score(y_test, model_pred_dtree)*100))
print("Confusion matrix: \n", confusion_matrix(y_test, model_pred_dtree))
print("Classification report: \n", classification_report(y_test, model_pred_dtree)) 
=======

>>>>>>> parent of 197e2a0 (Update NEW EDA AND ANALYSIS.py)

# %%
# KNN model 
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors = 10)
knnmodel.fit(X_train, y_train)
model_pred_knn = knnmodel.predict(x_popularity)                                                                                                 
print("Accuracy: {:.5f}%".format(knnmodel.score(x_popularity, y_popularity)*100)) 

# %%
# SVM
#from sklearn.svm import SVC
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
#svmodel = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#svmodel.fit(X_train, y_train)
#print(svmodel.score(X_test, y_test))

# %%
# MLP classifier model
from sklearn.neural_network import MLPClassifier
mlpmodel = MLPClassifier()
mlpmodel.fit(X_train, y_train)
model_pred_mlp = mlpmodel.predict(x_popularity)
print("Accuracy: {:.5f}%".format(mlpmodel.score(x_popularity, y_popularity)*100))

# %%
# Random forest model
from sklearn.ensemble import RandomForestClassifier
rfcmodel = RandomForestClassifier()
rfcmodel.fit(X_train, y_train)
model_pred_rfc = rfcmodel.predict(x_popularity)
print("Accuracy: {:.5f}%".format(rfcmodel.score(x_popularity, y_popularity)*100))

# %%
