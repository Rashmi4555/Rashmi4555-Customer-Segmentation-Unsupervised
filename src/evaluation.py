"""
Clustering evaluation module
"""
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ClusterEvaluator:
    """Evaluates clustering performance using multiple metrics"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def evaluate(self, X, labels, n_clusters):
        """Evaluate clustering performance"""
        if n_clusters < 2:
            return {
                'silhouette': -1,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0,
                'n_clusters': n_clusters
            }
        
        # Filter out noise points for DBSCAN
        valid_mask = labels != -1
        
        if valid_mask.sum() < 2:
            return {
                'silhouette': -1,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0,
                'n_clusters': n_clusters
            }
        
        X_valid = X[valid_mask] if -1 in labels else X
        labels_valid = labels[valid_mask] if -1 in labels else labels
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X_valid, labels_valid)
        except:
            silhouette = -1
        
        try:
            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
        except:
            davies_bouldin = float('inf')
        
        try:
            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
        except:
            calinski_harabasz = 0
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(X, labels, n_clusters)
        
        results = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'noise_percentage': ((labels == -1).sum() / len(labels) * 100) if -1 in labels else 0
        }
        
        return results
    
    def _calculate_cluster_statistics(self, X, labels, n_clusters):
        """Calculate statistics for each cluster"""
        stats = {}
        
        for cluster_id in range(n_clusters):
            if cluster_id == -1:  # Skip noise for DBSCAN
                continue
                
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size > 0:
                cluster_data = X[cluster_mask]
                
                stats[cluster_id] = {
                    'size': cluster_size,
                    'percentage': (cluster_size / len(X)) * 100,
                    'mean_values': cluster_data.mean(axis=0).tolist(),
                    'std_values': cluster_data.std(axis=0).tolist(),
                    'min_values': cluster_data.min(axis=0).tolist(),
                    'max_values': cluster_data.max(axis=0).tolist()
                }
        
        return stats
    
    def compare_algorithms(self, results_dict):
        """Compare multiple clustering algorithms"""
        comparison = {}
        
        for algo_name, result in results_dict.items():
            comparison[algo_name] = {
                'n_clusters': result['n_clusters'],
                'silhouette': result['scores']['silhouette'],
                'davies_bouldin': result['scores']['davies_bouldin'],
                'calinski_harabasz': result['scores']['calinski_harabasz'],
                'noise_percentage': result['scores'].get('noise_percentage', 0)
            }
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison).T
        
        # Rank algorithms
        df_comparison['silhouette_rank'] = df_comparison['silhouette'].rank(ascending=False)
        df_comparison['davies_bouldin_rank'] = df_comparison['davies_bouldin'].rank(ascending=True)
        df_comparison['calinski_harabasz_rank'] = df_comparison['calinski_harabasz'].rank(ascending=False)
        
        # Overall rank (average of individual ranks)
        df_comparison['overall_rank'] = df_comparison[
            ['silhouette_rank', 'davies_bouldin_rank', 'calinski_harabasz_rank']
        ].mean(axis=1)
        
        df_comparison = df_comparison.sort_values('overall_rank')
        
        return df_comparison
    
    def generate_report(self, results_dict, algorithm_names):
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # Compare algorithms
        comparison_df = self.compare_algorithms(results_dict)
        
        # Best algorithm
        best_algo = comparison_df.index[0]
        
        report['summary'] = {
            'best_algorithm': best_algo,
            'best_n_clusters': results_dict[best_algo]['n_clusters'],
            'best_silhouette': results_dict[best_algo]['scores']['silhouette'],
            'algorithm_ranking': comparison_df['overall_rank'].to_dict()
        }
        
        # Detailed results
        for algo_name in algorithm_names:
            if algo_name in results_dict:
                report['detailed_results'][algo_name] = results_dict[algo_name]['scores']
        
        # Recommendations
        best_result = results_dict[best_algo]
        n_clusters = best_result['n_clusters']
        
        if best_result['scores']['silhouette'] > 0.5:
            report['recommendations'].append(
                f"✅ Excellent clustering (Silhouette: {best_result['scores']['silhouette']:.3f})"
            )
        elif best_result['scores']['silhouette'] > 0.25:
            report['recommendations'].append(
                f"⚠️ Fair clustering (Silhouette: {best_result['scores']['silhouette']:.3f}) - Consider feature engineering"
            )
        else:
            report['recommendations'].append(
                f"❌ Poor clustering (Silhouette: {best_result['scores']['silhouette']:.3f}) - Algorithm or data may not be suitable"
            )
        
        report['recommendations'].append(
            f"Optimal number of clusters: {n_clusters}"
        )
        
        return report, comparison_df