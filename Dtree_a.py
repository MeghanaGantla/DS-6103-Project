# %%
# Dtree models

# %%
# importing necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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

# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# training a DescisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
