"""
DBSCAN Clustering Implementation
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DBSCANClustering:
    """DBSCAN clustering implementation with automatic parameter tuning"""
    
    def __init__(self, min_samples_range=(3, 10)):
        self.min_samples_range = min_samples_range
        self.model = None
        self.optimal_eps = None
        self.optimal_min_samples = None
        self.n_clusters = None
    
    def find_optimal_eps(self, X, k=4):
        """Find optimal epsilon using k-distance graph"""
        # Calculate k-distance
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        
        # Sort distances
        k_distances = np.sort(distances[:, k-1])
        
        # Find elbow point
        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)
        elbow_point = np.argmax(diffs2)
        
        self.optimal_eps = k_distances[elbow_point]
        self.k_distances = k_distances
        
        return self.optimal_eps
    
    def tune_parameters(self, X):
        """Tune DBSCAN parameters"""
        eps_values = []
        min_samples_values = []
        n_clusters_values = []
        silhouette_values = []
        
        # Find optimal eps first
        optimal_eps = self.find_optimal_eps(X)
        
        # Try different min_samples values
        for min_samples in range(self.min_samples_range[0], self.min_samples_range[1] + 1):
            dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Count clusters (excluding noise = -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:
                # Calculate silhouette (excluding noise points)
                valid_mask = labels != -1
                if valid_mask.sum() > 0:
                    sil_score = silhouette_score(X[valid_mask], labels[valid_mask])
                else:
                    sil_score = -1
            else:
                sil_score = -1
            
            eps_values.append(optimal_eps)
            min_samples_values.append(min_samples)
            n_clusters_values.append(n_clusters)
            silhouette_values.append(sil_score)
        
        # Find best parameters
        valid_indices = [i for i, s in enumerate(silhouette_values) if s > 0]
        
        if valid_indices:
            best_idx = valid_indices[np.argmax([silhouette_values[i] for i in valid_indices])]
            self.optimal_eps = eps_values[best_idx]
            self.optimal_min_samples = min_samples_values[best_idx]
            self.n_clusters = n_clusters_values[best_idx]
        else:
            # Default to reasonable values
            self.optimal_eps = optimal_eps
            self.optimal_min_samples = self.min_samples_range[0]
            self.n_clusters = 1
        
        return self.optimal_eps, self.optimal_min_samples
    
    def plot_k_distance(self, X, k=4):
        """Plot k-distance graph"""
        if not hasattr(self, 'k_distances'):
            self.find_optimal_eps(X, k)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.k_distances)), self.k_distances)
        plt.axhline(y=self.optimal_eps, color='r', linestyle='--', 
                   label=f'Optimal eps = {self.optimal_eps:.3f}')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-distance')
        plt.title('k-Distance Graph for DBSCAN')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def fit_predict(self, X):
        """Fit DBSCAN and predict clusters"""
        # Tune parameters
        if self.optimal_eps is None:
            self.tune_parameters(X)
        
        # Train model
        self.model = DBSCAN(
            eps=self.optimal_eps,
            min_samples=self.optimal_min_samples
        )
        
        labels = self.model.fit_predict(X)
        
        # Count clusters excluding noise
        unique_labels = set(labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Calculate noise percentage
        noise_points = (labels == -1).sum()
        self.noise_percentage = (noise_points / len(labels)) * 100
        
        print(f"  DBSCAN: {self.n_clusters} clusters, {self.noise_percentage:.1f}% noise")
        
        return labels, self.n_clusters
    
    def get_model_params(self):
        """Get model parameters"""
        return {
            'eps': self.optimal_eps,
            'min_samples': self.optimal_min_samples,
            'n_clusters': self.n_clusters,
            'noise_percentage': getattr(self, 'noise_percentage', 0)
        }