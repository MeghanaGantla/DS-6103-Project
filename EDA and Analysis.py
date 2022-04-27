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
# Adding a new column
print(df['song_popularity'].mean())
df["popularity"]= [ 1 if i >= 60 else 0 for i in df.song_popularity ]
print(df["popularity"].value_counts())

#%% 
# Checking datatypes 
df.info() 

#%%
# Describing data
df.describe().transpose()

# %%
# Removing irrelevant columns
df.drop(['song_name'], axis = 1, inplace = True)
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
nf = df.drop(["key", "mode", "tsign", "song_popularity", "popularity"], axis = 1)
nf.head()

#%% 
# Seperating Categorical data
cf = df[["key", "mode", "tsign", "popularity"]]
cf.head()

#%%
# Histograms for Numerical data
plt.figure(figsize=(25,25))
for i in range(len(nf.columns)):
    plt.subplot(5, 2, i+1)
    sns.distplot(nf[nf.columns[i]], color = list(np.random.randint([255,255,255])/255))
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
sns.countplot(x = "popularity", data = df, palette="Set2")
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
#see song_popularity as continuous variable
model = ols(formula='song_popularity ~ key + mode + tsign + duration + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
print( type(model) ) 

modelFit = model.fit()
print( modelFit.summary() )
#Adj. R-squared: 0.017
#%%
#Drop variables which P-value <0.05 which are 'duration' and 'key'
model2 = ols(formula='song_popularity ~ mode + tsign + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
print( type(model2) ) 
model2Fit = model2.fit()
print( model2Fit.summary() )
#Adj. R-squared:  0.017
#The linear model is not suit for this dataset 
#%%
#confirm the former conclusion
df["song_popularity"].where(df["song_popularity"] < 99, 99, True)
df['song_popularity'].value_counts()

x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['song_popularity']
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.sum(abs(y_pred - y_test)) / len(y_pred))
accu = model.score(X_test, y_test)
print("blah: {:.5f}%".format(accu * 100))
#1.5566%
#%%
#change the popularity from continuous as category variable

x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['popularity']
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.sum(abs(y_pred - y_test)) / len(y_pred))
accu = model.score(X_test, y_test)
print("blah: {:.5f}%".format(accu * 100))
#1.84596%

#The linear model not acceptable 
#%%
#consider logistic model 
#%%
#Then I change the data to build logistic model
#First I want to seperate the data into three part which are not-popular medium-popular and popular which are shown as 0, 1, 2 
#To do that I want to know the distribution of song_popular
# %%
sns.distplot(df["song_popularity"],color = list(np.random.randint([255,255,255])/255))

#%%
df["popularity_category"]= [ 2 if i >= 66 else 1 if i >=33 else 0 for i in df.song_popularity ]

# %%
#Use "popularity" as y to built logistic model 
from statsmodels.formula.api import glm
import statsmodels.api as sm 
modelLogit = glm(formula='popularity ~ C(mode) + C(tsign) + C(key) + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df, family=sm.families.Binomial())
modelLogitFit = modelLogit.fit()
print( modelLogitFit.summary() )


#%%
#Use sklearn package to simulate logistic model
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(f'lr train score:  {lr.score(X_train,y_train)}')
print(f'lr test score:  {lr.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, lr.predict(X_test)))
#%%
y_pred_probs=lr.predict_proba(X_test) 
#%%
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs[:, 1]) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

print("\nReady to continue.")
#%%
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


#%%

#%%
# modelpredicitons = pd.DataFrame(columns=['popularity'], data= modelLogitFit.predict(df))
# #%%
# cut_off = 0.3
# # Compute class predictions
# modelpredicitons['classLogitAll'] = np.where(modelpredicitons['popularity'] > cut_off, 1, 0)
# print(modelpredicitons.classLogitAll.head())
# #
# # Make a cross table
# print(pd.crosstab(df.popularity, modelpredicitons.classLogitAll,
# rownames=['Actual'], colnames=['Predicted'],
# margins = True))
# # Accuracy    = (TP + TN) / Total
# # Precision   = TP / (TP + FP)
# # Recall rate = TP / (TP + FN) = Sensitivity
# # Specificity = TN / (TN + FP)
# # F1_score is the "harmonic mean" of precision and recall
# # F1 = 2 (precision)(recall)/(precision + recall)

# %%
from statsmodels.formula.api import mnlogit
modelLogit2 = mnlogit(formula='popularity_category ~ C(mode) + C(tsign) + C(key) + acousticness + danceability + energy + liveness + loudness + speechiness + tempo + valence', data=df)
modelLogitFit2 = modelLogit2.fit()
print( modelLogitFit2.summary() )
#Pseudo R-squ: 0.01558

#%%Then bulid tree decision model

# %%
#Error: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
#So need change the part of dataframe
#df['popularity'].value_counts()
x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['song_popularity']
df["song_popularity"].where(df["song_popularity"] < 99, 99, True)
df['song_popularity'].value_counts()
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
#%%
dtree_popularity1 = DecisionTreeClassifier( random_state=1)
dtree_popularity1.fit(X_train,y_train)
y_test_pred = dtree_popularity1.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
#1.6%

#%%
x_popularity = df[['key', 'mode', 'tsign', 'duration', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
y_popularity = df['popularity']
df["popularity"].where(df["popularity"] < 99, 99, True)
df['popularity'].value_counts()
X_train, X_test, y_train, y_test= train_test_split(x_popularity, y_popularity, test_size=0.2, stratify=y_popularity,random_state=1)
#%%
dtree_popularity1 = DecisionTreeClassifier( random_state=1)
dtree_popularity1.fit(X_train,y_train)
y_test_pred = dtree_popularity1.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
#57.79%

#%%

# %%
#Then try knn model using two catelogy-popularity 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
y_pred = knn.predict(x_popularity )                                                                                                 
knn.score(x_popularity,y_popularity)
#73.99%

#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)
y_pred = knn.predict(x_popularity )                                                                                                 
knn.score(x_popularity,y_popularity)
#68.06%
print(confusion_matrix(y_test, y_pred))
#%%
from sklearn.metrics import classification_report
y_true, y_pred = y_test, knn.predict(x_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_test, y_pred))
#%%

