import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
 
np.random.seed(42)

n_samples = 1000

age = np.random.randint(18, 70, size=n_samples)  # Age between 18 and 70
annual_income = np.random.randint(25, 150, size=n_samples) * 1000  # Annual income in dollars
spending_score = np.random.randint(1, 101, size=n_samples)  # Spending score between 1 and 100

data = pd.DataFrame({
    'Age': age,
    'Annual Income (k$)': annual_income / 1000,  # Convert income to 'k$' (in thousands)
    'Spending Score (1-100)': spending_score
})

print(data.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

data['Cluster'] = kmeans.labels_

sil_score = silhouette_score(scaled_data, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

reduced_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
reduced_data['Cluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=reduced_data, palette='viridis', s=100, edgecolor='black')
plt.title('Customer Segments (Clustering Visualization)')
plt.show()

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

print("\nCluster Centers:")
print(centroid_df)

for i, row in centroid_df.iterrows():
    print(f"\nCluster {i + 1}:")
    print(f" - Average Age: {row['Age']}")
    print(f" - Average Annual Income: {row['Annual Income (k$)']}k$")
    print(f" - Average Spending Score: {row['Spending Score (1-100)']}")