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
