#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set environment variable to avoid KMeans memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Load the Dataset
data = pd.read_csv("Mall_Customers.csv")

# Fix column names (remove unwanted spaces)
data.columns = data.columns.str.strip()

# Check if "Gender" column exists
print("Columns in dataset:", data.columns)

# Step 2: Exploratory Data Analysis (EDA)
print("\nDataset Head:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSummary Statistics:")
print(data.describe())

# Visualize distributions
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(data['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='green')
plt.title('Annual Income Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True, color='red')
plt.title('Spending Score Distribution')

# Fix countplot issue
plt.subplot(2, 2, 4)
if 'Gender' in data.columns:
    sns.countplot(x=data['Gender'], palette='viridis')
    plt.title('Gender Distribution')
else:
    print("Column 'Gender' not found in dataset.")

plt.tight_layout()
plt.show()

# Step 3: Data Preprocessing
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine the Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply K-Means Clustering
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Step 6: Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segments: Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Step 7: Interpret the Clusters
# Analyze cluster characteristics
cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

# Add count of customers in each cluster
cluster_summary['Count'] = data['Cluster'].value_counts().sort_index()

print("\nCluster Summary:")
print(cluster_summary)


# In[ ]:




