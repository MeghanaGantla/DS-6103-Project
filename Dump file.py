#%%
sns.set()
fig, axes = plt.subplots(6, 2)
sns.histplot(data=nf, x='loudness', ax=axes[0,0])
sns.histplot(data=nf, x='liveness', ax=axes[0,1])
sns.histplot(data=nf, x='energy', ax=axes[1,0])
sns.countplot(data=cf, x='mode', ax=axes[1,1])

#%%
# %%
# Removing outliers
dfcopy = df.copy()
q_low = df["song_duration_ms"].quantile(0.25)
q_hi  = df["song_duration_ms"].quantile(0.75)

df_filtered = df[(df["song_duration_ms"] < q_hi) & (df["song_duration_ms"] > q_low)]


sns.histplot(df["song_duration_ms"], color="green")
plt.show()
sns.histplot(df["acousticness"], color = "blue")
plt.show()
plt.subplot()
sns.boxplot(x = "song_duration_ms", color = "green", data=df)
plt.show()

#%% 
# Histograms for numerical data
nf.hist(layout=(5,2), figsize=(25,25), color = list(np.random.randint([255,255,255])/255))
plt.show()
# %%
# Count plots for categorical data
cf.hist(layout = (3,1), figsize=(25,25))
plt.show()

#%%
def get_outlier_counts(df, threshold):
    df = df.copy()
    
    # Get the z-score for specified threshold
    threshold_z_score = stats.norm.ppf(threshold)
    
    # Get the z-scores for each value in df
    z_score_df = pd.DataFrame(np.abs(stats.zscore(df)), columns=df.columns)
    
    # Compare df z_scores to the threshold and return the count of outliers in each column
    return (z_score_df > threshold_z_score).sum(axis=0)

#%%
get_outlier_counts(df, 0.99999999999)

#%%
def remove_outliers(df, threshold):
    df = df.copy()
    
    # Get the z-score for specified threshold
    threshold_z_score = stats.norm.ppf(threshold)
    
    # Get the z-scores for each value in df
    z_score_df = pd.DataFrame(np.abs(stats.zscore(df)), columns=df.columns)
    z_score_df = z_score_df > threshold_z_score
    
    # Get indices of the outliers
    outliers = z_score_df.sum(axis=1)
    outliers = outliers > 0
    outlier_indices = df.index[outliers]
    
    # Drop outlier examples
    df = df.drop(outlier_indices, axis=0).reset_index(drop=True)
    
    return df

#%%

def preprocess_inputs(df, outliers=True, threshold=0.95):
    df = df.copy()
    
    # Remove outliers if specified
    if outliers == False:
        df = remove_outliers(df, threshold)
    
    # Split df into X and y
    y = df['popularity'].copy()
    X = df.drop('popularity', axis=1).copy()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
    # Scale X with a standard scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

#%%
outlier_X_train, outlier_X_test, outlier_y_train, outlier_y_test = preprocess_inputs(df, outliers=True)

X_train, X_test, y_train, y_test = preprocess_inputs(df, outliers=False, threshold=0.8)

#%%
# With outliers

outlier_model = LinearRegression()
outlier_model.fit(outlier_X_train, outlier_y_train)

outlier_model_acc = outlier_model.score(outlier_X_test, outlier_y_test)

print("Test Accuracy (Outliers): {:.5f}%".format(outlier_model_acc * 100))


#%%
# Without outliers

model = LinearRegression()
model.fit(X_train, y_train)

model_acc = model.score(X_test, y_test)

print("Test Accuracy (No Outliers): {:.5f}%".format(model_acc * 100))
# %%
#_____________________________________________________

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
# __________________________________________ 03:02am 05/26
#%%
from sklearn.preprocessing import OneHotEncoder
pd.get_dummies(df["key"])
#onehot = OneHotEncoder(categorical_features = [])


#%%
#Converting categorical Columns to Numeric
nvc = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values']/df.shape[0],3)*100
print(nvc)

df3 = df.copy()

ecc = nvc[nvc['Percentage']!=0].index.values
fcc = [i for i in cf if i not in ecc]
#One-Hot Binay Encoding
oh=True
dm=True
for i in fcc:
    #print(i)
    if df3[i].nunique()==2:
        if oh==True: print("\033[1mOne-Hot Encoding on features:\033[0m")
        print(i);oh=False
        df3[i]=pd.get_dummies(df3[i], drop_first=True, prefix=str(i))
    if (df3[i].nunique()>2 and df3[i].nunique()<17):
        if dm==True: print("\n\033[1mDummy Encoding on features:\033[0m")
        print(i);dm=False
        df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df3[i], drop_first=True, prefix=str(i)))],axis=1)
        
df3.shape
# %%
df1 = df3.copy()

#features1 = [i for i in features if i not in ['CHAS','RAD']]
features1 = nf

for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
display(df1.head())
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(df3.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))
# %%
df1.drop('song_popularity', axis=1)
y = df1['popularity'].copy()
X = df1.drop('popularity', axis=1).copy()
    
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
    
# %%
model = LinearRegression()
model.fit(X_train, y_train)

model_acc = model.score(X_test, y_test)

print("Test Accuracy (No Outliers): {:.5f}%".format(model_acc * 100))
# %%