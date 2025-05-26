#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import necessary libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[3]:


# Load the dataset
data = pd.read_csv('country_data.csv')

# Drop the 'country' column since it's non-numeric
data_features = data.drop(columns=['country'])


# In[4]:


#View data
data_features


# In[5]:


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)


# In[6]:


# Perform K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=5)
kmeans.fit(data_scaled)

# Get the cluster centroids and predictions
centers = kmeans.cluster_centers_
labels = kmeans.predict(data_scaled)


# In[7]:


# View the cluster centroids and assign labels (Developed, Developing, Underdeveloped)
print('Cluster Centroids:', centers)
cluster_means = pd.DataFrame(data_scaled).groupby(labels).mean()
print('\nCluster Means:\n', cluster_means)


# In[8]:


# Visualize the clusters in a 2D space
fig, ax = plt.subplots()

# Store the normalization of the color encodings based on the number of clusters
nm = Normalize(vmin=0, vmax=len(centers)-1)

# Plot the clustered data, using the first two features 
ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=50, cmap='plasma', norm=nm)

# Plot the cluster centroids and label them
for i in range(centers.shape[0]):
    if i == 0:
        label = "C0:Developed"
    elif i == 1:
        label = "C1:Underdeveloped"
    else:
        label = "C2:Developing"
    
    # Plot the centroid with labels
    ax.text(centers[i, 0], centers[i, 1], label, c='black', 
            bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

ax.set_xlabel('Feature 1 (Standardized)')
ax.set_ylabel('Feature 2 (Standardized)')
ax.set_title('Country Clustering by K-Means')

plt.show()


# ## Author: Ajakaiye, Oluwadamilola Oreofe
# ### Data Science Consultant
# ### 20th June, 2024
