"""
Hierarchical Clustering Implementation
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClustering:
    """Hierarchical clustering implementation"""
    
    def __init__(self, max_clusters=10, linkage_method='ward'):
        self.max_clusters = max_clusters
        self.linkage_method = linkage_method
        self.model = None
        self.optimal_k = None
        self.linkage_matrix = None
    
    def find_optimal_k(self, X):
        """Find optimal number of clusters using silhouette score"""
        silhouette_scores = []
        
        for k in range(2, self.max_clusters + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage_method
            )
            labels = model.fit_predict(X)
            
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Choose K with best silhouette score
        self.optimal_k = np.argmax(silhouette_scores) + 2
        self.silhouette_scores = silhouette_scores
        
        return self.optimal_k
    
    def create_dendrogram(self, X, max_display=100):
        """Create dendrogram visualization"""
        if len(X) > max_display:
            # Sample for faster dendrogram
            indices = np.random.choice(len(X), max_display, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        self.linkage_matrix = linkage(X_sample, method=self.linkage_method)
        
        plt.figure(figsize=(12, 6))
        dendrogram(self.linkage_matrix)
        plt.title(f'Dendrogram ({self.linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.axhline(y=self.linkage_matrix[-self.optimal_k, 2], 
                   color='r', linestyle='--', label=f'Optimal K={self.optimal_k}')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def fit_predict(self, X):
        """Fit hierarchical clustering and predict clusters"""
        # Find optimal K
        if self.optimal_k is None:
            self.find_optimal_k(X)
        
        # Train model
        self.model = AgglomerativeClustering(
            n_clusters=self.optimal_k,
            linkage=self.linkage_method
        )
        
        labels = self.model.fit_predict(X)
        
        return labels, self.optimal_k
    
    def get_model_params(self):
        """Get model parameters"""
        return {
            'n_clusters': self.optimal_k,
            'linkage_method': self.linkage_method,
            'silhouette_scores': getattr(self, 'silhouette_scores', None)
        }