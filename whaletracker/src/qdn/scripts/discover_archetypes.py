"""
Archetype Discovery Script (The Elbow Phase)
Uses K-Means to find hidden structural archetypes in 10-year data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchetypeDiscoverer:
    def __init__(self, data_path: str):
        self.raw_data = pd.read_csv(data_path)
        # Assuming 30 features + 'ticker' + 'forward_return'
        self.features = self.raw_data.iloc[:, :30] 
        self.returns = self.raw_data['forward_return']
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

    def find_optimal_k(self, max_k=20):
        """
        Implements the Elbow Method and Silhouette Analysis.
        """
        inertias = []
        silhouettes = []
        K = range(2, max_k + 1)

        logger.info("Calculating Elbow and Silhouette for K=2 to 20...")
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_features)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.scaled_features, kmeans.labels_))

        # Automate "Elbow" detection or let user decide via plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(K, inertias, 'bo-')
        plt.xlabel('k (Number of Archetypes)')
        plt.ylabel('Inertia (Error)')
        plt.title('The Elbow Method')

        plt.subplot(1, 2, 2)
        plt.plot(K, silhouettes, 'ro-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig('archetype_discovery_metrics.png')
        logger.info("Metrics saved to archetype_discovery_metrics.png")

    def train_final_clusters(self, k: int):
        """
        Once K is decided, group and audit performance.
        """
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.scaled_features)
        
        df = self.raw_data.copy()
        df['cluster_id'] = clusters
        
        # Audit: Which cluster has the most "Boring Gems"?
        audit = df.groupby('cluster_id').agg({
            'forward_return': ['mean', 'median', 'std'],
            'ticker': 'count'
        })
        
        # Convexity = Mean / Std (Looking for high reward/risk)
        audit['convexity'] = audit[('forward_return', 'mean')] / audit[('forward_return', 'std')]
        
        logger.info("\n--- Archetype Audit ---")
        print(audit.sort_values('convexity', ascending=False))
        
        # Save centroids for DQN training
        centroids = kmeans.cluster_centers_
        np.save('structural_centroids.npy', centroids)
        logger.info("Structural centroids saved for DQN.")

if __name__ == "__main__":
    # Placeholder for actual data path
    # discoverer = ArchetypeDiscoverer("data/historical_features_10y.csv")
    # discoverer.find_optimal_k()
    print("Script ready. Awaiting 10-year dataset to run the Elbow Method.")
