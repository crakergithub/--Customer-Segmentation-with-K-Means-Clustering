# Project Title: Customer Segmentation with K-Means Clustering
# Internship Domain: Data Science
# Project Level: Entry Level
# Assigned By: CodeClause Internship
# Assigned To: [Your Name]
# Start Date: [Start Date]
# End Date: [End Date]

# Project Aim: Apply K-Means clustering to segment customers based on their purchase behavior.
# Description: Use a customer purchase dataset to identify distinct segments using the K-Means clustering algorithm.
# Technologies: Python, Pandas, Scikit-learn.

# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/purchase behaviou dataset.csv')

# Displaying the first few rows of the dataset
print(data.head())

# Preprocessing the data (if needed)
# Here you may need to handle missing values, encode categorical variables, scale the features, etc.

# Selecting the relevant features for clustering
X = data[['LIFESTAGE', 'PREMIUM_CUSTOMER']]

# Perform one-hot encoding on categorical columns
X_encoded = pd.get_dummies(X)

# Choosing the number of clusters (You can use techniques like Elbow Method or Silhouette Score)
# For simplicity, let's assume the number of clusters to be 3
k = 3

# Applying K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_encoded)

# Adding cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Visualizing the clusters (2D example)
# Since the features are not numeric, visualization might not be straightforward
# You can explore visualization techniques suitable for categorical data
# For simplicity, we'll skip visualization in this example

# Printing cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Printing the count of data points in each cluster
print("\nCount of data points in each cluster:")
print(data['Cluster'].value_counts())

# You can further analyze each cluster to understand the characteristics of customers in each segment
