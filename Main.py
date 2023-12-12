# -*- coding: utf-8 -*-
"""M22EE051_Task.ipynb

# a.) Perform PCA on MNIST and then perform GMM clustering. (Library can be used for SVD and GMM) but PCA should be from scratch. PCA should be done for 32, 64 and 128 components.Clustering should be done in 10, 7, and 4 clusters.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# Function to perform PCA from scratch
def pca(X, n_components):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project data onto principal components
    X_pca = np.dot(X_centered, Vt.T[:, :n_components])

    return X_pca

# Function to perform GMM clustering with adjusted parameters
def gmm_clustering(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10, max_iter=500, tol=1e-3)
    gmm.fit(X)
    return gmm.predict(X)

# Function to visualize images in clusters
def visualize_clusters(images, labels, clusters, num_clusters):
    plt.figure(figsize=(12, 8))
    for i in range(min(10, num_clusters)):
        cluster_indices = np.where(clusters == i)[0]
        for j in range(min(5, len(cluster_indices))):
            plt.subplot(5, num_clusters, j * num_clusters + i + 1)
            reshaped_image = images[cluster_indices[j]].reshape(28, 28) #reshaping image to original size 28*28 for visualization
            plt.imshow(reshaped_image, cmap='gray')
            plt.title(f'Cluster {i}')
            plt.axis('off')
    plt.show()

"""# b.) Visualize the images getting clustered in different clusters."""

# Load the MNIST dataset
mnist = np.loadtxt('/content/mnist_train.csv', delimiter=',', skiprows=1)

# Extract labels and features
labels = mnist[:, 0].astype(int)
features = mnist[:, 1:]

# Normalize features
features_normalized = features / 255.0


# Apply PCA and GMM clustering for different components and clusters

components_list = [32, 64, 128]
clusters_list = [10, 7, 4]

for n_components in components_list:
    # Perform PCA
    print("PCA on image data of 784 dimension to ",n_components,"dimension")
    features_pca = pca(features_normalized, n_components)
    print("Size of data after PCA:", features_pca.shape)
    for n_clusters in clusters_list:
        # Perform GMM clustering with adjusted parameters
        clusters_gmm = gmm_clustering(features_pca, n_clusters)

        print("GMM clustering of images to ",n_clusters,"clusters")

        # Visualize the images in different clusters
        visualize_clusters(features_normalized, labels, clusters_gmm, n_clusters)

"""# d.) Can you find the optimal number of components the PCA should choose which covers almost all the necessary patterns in the data? Can you comment on where PCA can fail?"""

def find_optimal_pca_components(X):
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    explained_variance_ratio = eigenvalues_sorted / eigenvalues_sorted.sum()
    return explained_variance_ratio

explained_variance_ratio = find_optimal_pca_components(features_normalized)

# Plot explained variance ratio
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.show()
