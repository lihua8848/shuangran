import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the data
data = pd.read_csv('/mnt/data/each_slide_patch_num_label_vitb.csv')
data.columns = ['data1', 'data2', 'data3', 'data4', 'label']

# Descriptive statistics
desc_stats = data.describe()

# Plotting descriptive statistics
desc_stats.drop(['count']).plot(kind='bar', figsize=(15, 7), title="Descriptive Statistics")
plt.ylabel('Value')
plt.show()

# Correlation analysis
correlation_matrix = data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Splitting the data into training and testing sets
X = data[['data1', 'data2', 'data3', 'data4']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot histograms for each feature
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
fig.suptitle('Distribution of Features', fontsize=16)

sns.histplot(data=data, x='data1', kde=True, ax=axes[0,0])
axes[0,0].set_title('Distribution of data1')

sns.histplot(data=data, x='data2', kde=True, ax=axes[0,1])
axes[0,1].set_title('Distribution of data2')

sns.histplot(data=data, x='data3', kde=True, ax=axes[1,0])
axes[1,0].set_title('Distribution of data3')

sns.histplot(data=data, x='data4', kde=True, ax=axes[1,1])
axes[1,1].set_title('Distribution of data4')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
