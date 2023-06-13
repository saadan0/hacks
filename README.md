start here
nn2 = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Assuming your dataframe has features in columns 'feature1', 'feature2', ..., and the target label in 'target'
X = df.drop('target', axis=1).values
y = df['target'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model architecture
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))  # First hidden layer with ReLU activation
model.add(Dense(16, activation='tanh'))  # Second hidden layer with tanh activation
model.add(Dense(8, activation='relu'))  # Third hidden layer with ReLU activation
model.add(Dense(4, activation='tanh'))  # Fourth hidden layer with tanh activation
model.add(Dense(2, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)[:, 1]  # Probability of positive class for ROC curve
y_pred = np.argmax(model.predict(X_test), axis=1)  # Predicted labels for other metrics

# Compute metrics
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred_prob)
roc_auc = roc_auc_score(y_test[:, 1], y_pred_prob)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
confusion_mat = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
classification_rep = classification_report(np.argmax(y_test, axis=1), y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot accuracy curve
epoch_nums = range(1, len(model.history.history['accuracy']) + 1)
plt.plot(epoch_nums, model.history.history['accuracy'], label='Training Accuracy')
plt.plot(epoch_nums, model.history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Display metrics
print('Accuracy: {:.4f}'.format(accuracy))
print('Classification Report:')
print(classification_rep)

'''
nn = '''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

X_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=10))
model.add(Dense(16, activation='tanh')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(2, activation='softmax'))  

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
X_test = np.random.random((100, 10))
predictions = model.predict(X_test)
print(predictions)
'''
lr='''

#generate some random regression data
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#single line plot
plt.scatter(X_test, y_test, color='b', label='Actual')
plt.plot(X_test, y_pred, color='r', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()

#seperate plots
df_plot = pd.concat([X_train, y_train], axis=1)
sns.pairplot(df_plot, x_vars=X_train.columns, y_vars=target_column, kind='reg')
plt.show()

#making 3d plot for 3 features
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, color='b', label='Actual')
ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_pred, color='r', label='Predicted')
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('target')
ax.set_title('Multi-Regression Line')
ax.legend()
plt.show()

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
#with stats models
x = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print(f"coefficient of determination: {results.rsquared}")
print(f"adjusted coefficient of determination: {results.rsquared_adj}")
print(f"regression coefficients: {results.params}")

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred.round())
plt.imshow(confusion_mat, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(y_test, y_pred)

# Calculate the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Plot accuracy curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(accuracy, label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend(loc='lower right')
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

#errors
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))

'''
kmeans='''

df.drop(['label'], axis = 1, inplace =True)
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
kmeans_blob = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans_blob = kmeans_blob.fit_predict(df_blob_kmeans)
df_blob_kmeans['Cluster'] = y_kmeans_blob

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,30))
fig.suptitle('ANSWER vs K-Means
', size = 18)
axes[0,0].scatter(blob_df['x'], blob_df['y'], c=blob_df['color'], s=10, cmap = "Set3")
axes[0,0].set_title("Answer Blob");
axes[0,1].scatter(df_blob_kmeans['x'], df_blob_kmeans['y'], c=df_blob_kmeans['Cluster'], s=10, cmap = "Set3")
axes[0,1].set_title("K-Means Blob");

'''
elbow='''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the iris dataset
iris = load_iris()
X = iris.data
min_clusters = 1
max_clusters = 10
wcss = []
for num_clusters in range(min_clusters, max_clusters+1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(min_clusters, max_clusters+1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Curve')
plt.show()

'''
silvt='''

# Select the two columns
X = df[['x','y']]
# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

range_n_clusters = range(2, 10)
score = []
clusters = []
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    clusters.append(n_clusters)
    score.append(silhouette_avg)
    print("For n_clusters = {}, the silhouette score is {:.2f}".format(n_clusters, silhouette_avg))
res = {"clusters":clusters,"silhouette score":score}
res = pd.DataFrame(res)
res

'''
elbow2='''

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)

print(kl.elbow)

'''
silvt2='''

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

'''
kscore='''

ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)

'''
eda='''

data.head(5)
data.describe()
data.drop(['id'], axis=1, inplace=True)
data.isnull().sum()
data.dtypes
# Fill missing values in column 'B' with the mean
mean_value = df['B'].mean()
df['B'] = df['B'].fillna(mean_value)
median_value = df['C'].median()
df['C'] = df['C'].fillna(median_value)
mode_value = df['A'].mode()[0]
df['A'] = df['A'].fillna(mode_value)
df['B'] = df['B'].replace('apple', 'orange')
df['A'] = df['A'].astype(float)

'''
outlier='''

# Loop over each column in the dataframe
for column in data.columns:
  if data[column].dtype == 'int64' or data[column].dtype == 'float64':
    # Calculate the 1st and 3rd quartiles of the column
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count the number of outliers in the column
    num_outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
    print(f'{column} has {num_outliers} outliers')

    # Create a boxplot of the column if there are outliers
    if num_outliers > 0:
        data.boxplot(column=[column])
        plt.title(f'Boxplot of {column}')
        plt.show()
        # Remove the outliers from the column
        #data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

'''
corr='''

# Create correlation matrix
corr_matrix = data.corr()

# Visualize correlation matrix with heatmap
sns.heatmap(corr_matrix, annot=True)

# Remove columns with high covariance
cols_to_drop = []
for col in corr_matrix.columns:
    for idx, val in corr_matrix[col].iteritems():
        if idx != col and abs(val) > 0.95: #i have increased the correation value becuase we have less data
            if col not in cols_to_drop:
                cols_to_drop.append(col)

# Remove columns from dataframe
data.drop(cols_to_drop, axis=1, inplace = True)
print(cols_to_drop)

'''
pca='''

df = copy.deepcopy(data)

df.drop(['diagnosis'], axis=True, inplace=True)

# Define which columns are numeric and which are categorical
numeric_columns = list(df.select_dtypes(include=['int64', 'float64']).columns)
categorical_columns = list(df.select_dtypes(include=['object']).columns)

# Create transformers for both numeric and categorical columns
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Combine the transformers using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Fit and transform the DataFrame
df_preprocessed = preprocessor.fit_transform(df)

pca = PCA()
pca.fit(df_preprocessed)

# Calculate the cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

# Determine the optimal number of components
threshold = 0.95  # e.g., 95% of the total variance
optimal_n_components = np.where(cumulative_explained_variance >= threshold)[0][0] + 1

print(f"Optimal number of components: {optimal_n_components}")
# PCA is not reducing the number of components so there is no need to apply it on our dataset
df = copy.deepcopy(data)
# Separate the target variable from the features
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Standardize the feature data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply PCA to reduce dimensions
n_components = 10
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_standardized)

# Convert the result back to a pandas DataFrame
X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Reattach the target variable
df_reduced = pd.concat([X_reduced_df, y], axis=1)

print(df_reduced)

'''
pca2='''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your data into a pandas DataFrame
# (Replace this with your actual data)
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'feature3': [10, 20, 30, 40, 50],
        'class': ['A', 'B', 'A', 'A', 'B']}
df = pd.DataFrame(data)

# Separate the target variable from the features
X = df.drop(columns=['class'])
y = df['class']

# Standardize the feature data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply PCA to reduce dimensions
n_components = 2
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_standardized)

# Convert the result back to a pandas DataFrame
X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Reattach the target variable
df_reduced = pd.concat([X_reduced_df, y], axis=1)

print(df_reduced)

'''
norm='''

data = copy.deepcopy(df_reduced)
#normalization
num_cols = data.select_dtypes(include=['int64', 'float64', 'int32']).columns
# Apply MinMaxScaler
minmax = MinMaxScaler()
data[num_cols] = minmax.fit_transform(data[num_cols])
data[num_cols]

'''
onehot='''

import pandas as pd

# Create a sample dataframe with a categorical column
data = {
    'Feature1': ['A', 'B', 'B', 'A', 'C', 'C', 'B', 'A', 'C', 'C']
}

df = pd.DataFrame(data)

# Perform one-hot encoding on the 'Feature1' column
encoded_df = pd.get_dummies(df['Feature1'], prefix='Feature1')

# Concatenate the original dataframe with the encoded columns
df_encoded = pd.concat([df, encoded_df], axis=1)

# Print the encoded dataframe
print(df_encoded)

'''
hist='''

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
})
df.hist()
plt.show()
for column in df.columns:
    sns.kdeplot(df[column])
    plt.xlabel(column)
    plt.show()

'''
plots='''

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
})

# Line plot using Matplotlib
plt.plot(df['A'], label='A')
plt.plot(df['B'], label='B')
plt.plot(df['C'], label='C')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.show()

# Scatter plot using Matplotlib
plt.scatter(df['A'], df['B'])
plt.xlabel('A')
plt.ylabel('B')
plt.title('Scatter Plot')
plt.show()

# Bar plot using Matplotlib
df.plot(kind='bar')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Bar Plot')
plt.show()

# Box plot using Plotly
fig = px.box(df, title='Box Plot')
fig.show()

# Heatmap using Plotly
fig = go.Figure(data=go.Heatmap(z=df.values, x=df.columns, y=df.index))
fig.show()

# 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='A', y='B', z='C', title='3D Scatter Plot')
fig.show()

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10],
    'C': [11, 12, 13, 14, 15]
})

# Plot Probability Density Function (PDF)
for column in df.columns:
    sns.kdeplot(df[column], label=column)

plt.title('PDF Plots')
plt.legend()
plt.show()

# Plot Cumulative Distribution Function (CDF)
for column in df.columns:
    sns.kdeplot(df[column], cumulative=True, label=column)

plt.title('CDF Plots')
plt.legend()
plt.show()

'''
datasets='''

dataset_names = datasets.load_boston(return_X_y=False).keys()
print(dataset_names)
# Load the dataset
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

'''
neural='''

df = pd.read_csv('your_data.csv')

# Preprocess the data
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
#history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Plot the training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the classification report
report = classification_report(y_true, y_pred_classes)
print(report)

from sklearn.metrics import roc_curve, auc

y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

'''
end here
