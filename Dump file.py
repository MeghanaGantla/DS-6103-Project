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