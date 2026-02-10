"""
Gaussian Mixture Model Implementation
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

class GMMClustering:
    """Gaussian Mixture Model implementation with BIC/AIC selection"""
    
    def __init__(self, max_components=10, random_state=42):
        self.max_components = max_components
        self.random_state = random_state
        self.model = None
        self.optimal_components = None
        self.bic_scores = []
        self.aic_scores = []
    
    def find_optimal_components(self, X):
        """Find optimal number of components using BIC and AIC"""
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n in range(1, self.max_components + 1):
            gmm = GaussianMixture(
                n_components=n,
                random_state=self.random_state,
                covariance_type='full'
            )
            gmm.fit(X)
            
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            
            # Calculate silhouette score
            labels = gmm.predict(X)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow in BIC (lower is better)
        bic_diffs = np.diff(bic_scores)
        bic_elbow = np.argmax(np.diff(bic_diffs)) + 2
        
        # Find best silhouette score
        best_silhouette = np.argmax(silhouette_scores) + 1
        
        # Choose optimal components (prioritize silhouette)
        self.optimal_components = best_silhouette
        
        self.bic_scores = bic_scores
        self.aic_scores = aic_scores
        self.silhouette_scores = silhouette_scores
        self.bic_elbow = bic_elbow
        self.best_silhouette = best_silhouette
        
        return self.optimal_components
    
    def plot_information_criteria(self):
        """Plot BIC and AIC scores"""
        if not self.bic_scores:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # BIC Plot
        ax1.plot(range(1, len(self.bic_scores) + 1), self.bic_scores, 
                marker='o', color='blue', label='BIC')
        ax1.axvline(x=self.bic_elbow, color='r', linestyle='--',
                   label=f'BIC Elbow: K={self.bic_elbow}')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('BIC Score')
        ax1.set_title('Bayesian Information Criterion (BIC)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # AIC Plot
        ax2.plot(range(1, len(self.aic_scores) + 1), self.aic_scores,
                marker='o', color='green', label='AIC')
        ax2.axvline(x=self.optimal_components, color='r', linestyle='--',
                   label=f'Optimal: K={self.optimal_components}')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('AIC Score')
        ax2.set_title('Akaike Information Criterion (AIC)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def fit_predict(self, X):
        """Fit GMM and predict clusters"""
        # Find optimal components
        if self.optimal_components is None:
            self.find_optimal_components(X)
        
        # Train model
        self.model = GaussianMixture(
            n_components=self.optimal_components,
            random_state=self.random_state,
            covariance_type='full'
        )
        
        self.model.fit(X)
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        self.labels = labels
        self.probabilities = probabilities
        
        return labels, self.optimal_components
    
    def get_probability_matrix(self):
        """Get cluster membership probabilities"""
        if hasattr(self, 'probabilities'):
            return self.probabilities
        return None
    
    def get_model_params(self):
        """Get model parameters"""
        return {
            'n_components': self.optimal_components,
            'bic_scores': self.bic_scores,
            'aic_scores': self.aic_scores,
            'silhouette_scores': self.silhouette_scores,
            'converged': self.model.converged_ if self.model else None
        }