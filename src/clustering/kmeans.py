"""
K-Means Clustering Implementation
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class KMeansClustering:
    """K-Means clustering implementation with automatic K selection"""
    
    def __init__(self, max_clusters=10, random_state=42):
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.model = None
        self.optimal_k = None
        self.scores = {}
    
    def find_optimal_k(self, X):
        """Find optimal number of clusters using elbow method and silhouette"""
        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []
        
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if len(np.unique(kmeans.labels_)) > 1:
                sil_score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point
        wcss_diff = np.diff(wcss)
        wcss_diff_diff = np.diff(wcss_diff)
        elbow_point = np.argmax(wcss_diff_diff) + 3  # +3 due to double diff
        
        # Find best silhouette score
        best_silhouette = np.argmax(silhouette_scores) + 2
        
        # Choose optimal K (prioritize silhouette)
        self.optimal_k = best_silhouette
        
        self.scores['wcss'] = wcss
        self.scores['silhouette_scores'] = silhouette_scores
        self.scores['elbow_point'] = elbow_point
        self.scores['best_silhouette'] = best_silhouette
        
        return self.optimal_k
    
    def fit_predict(self, X):
        """Fit K-Means and predict clusters"""
        # Find optimal K if not already found
        if self.optimal_k is None:
            self.find_optimal_k(X)
        
        # Train model with optimal K
        self.model = KMeans(
            n_clusters=self.optimal_k,
            random_state=self.random_state,
            n_init=20,
            max_iter=300
        )
        
        labels = self.model.fit_predict(X)
        
        return labels, self.optimal_k
    
    def get_cluster_centers(self):
        """Get cluster centers"""
        if self.model is not None:
            return self.model.cluster_centers_
        return None
    
    def get_model_params(self):
        """Get model parameters"""
        return {
            'n_clusters': self.optimal_k,
            'inertia': self.model.inertia_ if self.model else None,
            'scores': self.scores
        }