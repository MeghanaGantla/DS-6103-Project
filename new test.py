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
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

#%%
# Reading data
df = pd.read_csv("SpotifyAudioFeaturesNov2018.csv")
df.head()

# %%
# # Changing column names
df.rename(columns = {'duration_ms':'duration', 'time_signature':'tsign', 'popularity':'song_popularity'}, inplace = True)
df.head()

#%%
# Adding a new column
print(df['song_popularity'].mean())
df["popularity"]= [1 if i >= 50 else 0 for i in df.song_popularity ]
#df["popularity"]= [2 if i >= 70 else 1 if (i>=50) & (i<=69) else 0 for i in df.song_popularity ]
print(df["popularity"].value_counts())

# %%
# Removing irrelevant columns
df.drop(["song_popularity"], axis = 1, inplace=True)
df.drop(["track_id"], axis = 1, inplace=True)
df.drop(["artist_name"], axis = 1, inplace=True)
df.drop(["track_name"], axis = 1, inplace=True)
df.head()

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
# Seperating Numerical data
nf = df.drop(["key", "mode", "tsign", "popularity"], axis = 1)
nf.head()

#%%
# Seperating categorical features
cf = df[["key", "mode", "tsign", 'popularity']]
cf.head()

#%%
# Converting categorical Columns to Numeric
nulls = pd.DataFrame(df.isnull().sum().sort_values(), columns=['null_values'])
print(nulls)
dupdf = df.copy()
e = nulls[nulls['null_values']!=0].index.values
f = [i for i in cf if i not in e]

# One-Hot Binay Encoder
oh=True
dm=True
for i in f:
    if dupdf[i].nunique() == 2:
        if oh==True: 
            print("One-Hot Encoding on features:")
        print(i) 
        oh=False
        dupdf[i]=pd.get_dummies(dupdf[i], drop_first=True, prefix=str(i))
    if (dupdf[i].nunique() > 2 and dupdf[i].nunique() < 17):
        if dm==True: 
            print("Dummy Encoding on features:")
        print(i)
        dm=False
        dupdf = pd.concat([dupdf.drop([i], axis=1), pd.DataFrame(pd.get_dummies(dupdf[i], drop_first=True, prefix=str(i)))],axis=1)     
dupdf.head()

#%%
# Label Encoder
from sklearn import preprocessing
dupdf = df.copy()
encoder = preprocessing.LabelEncoder()
dupdf["key"] = encoder.fit_transform(dupdf["key"])
encoder = preprocessing.LabelEncoder()
dupdf["mode"] = encoder.fit_transform(dupdf["mode"])
encoder = preprocessing.LabelEncoder()
dupdf["tsign"] = encoder.fit_transform(dupdf["tsign"])
encoder = preprocessing.LabelEncoder()
dupdf["popularity"] = encoder.fit_transform(dupdf["popularity"])
dupdf.head()

# %% Removing outliers

df1 = df.copy()
features1 = nf

for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(dupdf.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))
df1.head()
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
accu = model.score(X_test, y_test)
print("Accuracy: {:.5f}%".format(accu * 100))

#%%Then bulid tree decision model
#Error: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
#So need change the part of dataframe
#df['popularity'].value_counts()
#x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
#y_popularity = df['popularity']
#df["popularity"].where(df["popularity"] < 99, 99, True)
#df['popularity'].value_counts()
#X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
#%%
# Decision tree
dtree_popularity1 = DecisionTreeClassifier( random_state=1)
dtree_popularity1.fit(X_train,y_train)
y_test_pred = dtree_popularity1.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# %%
# KNN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
y_pred = knn.predict(x_popularity )                                                                                                 
knn.score(x_popularity, y_popularity)

# %%
