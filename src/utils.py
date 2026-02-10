"""
Utility functions for the customer segmentation project
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def save_results(data, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    data = convert_types(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")

def create_visualizations(X_pca, labels, customer_features, algorithm_name):
    """Create comprehensive visualizations"""
    print(f"Creating visualizations for {algorithm_name}...")
    
    # 1. 2D Scatter Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', 
                         alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Customer Segments - {algorithm_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/cluster_plots/{algorithm_name}_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 3D Plot (if enough dimensions)
    if X_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter_3d = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                               c=labels, cmap='tab20', alpha=0.7, s=50)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title(f'Customer Segments - {algorithm_name} (3D)')
        plt.colorbar(scatter_3d, ax=ax, label='Cluster')
        plt.savefig(f'results/cluster_plots/{algorithm_name}_3d.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    bars = plt.bar(range(len(unique_labels)), counts, color=colors, edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.title(f'Cluster Size Distribution - {algorithm_name}')
    plt.xticks(range(len(unique_labels)), unique_labels)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'results/cluster_plots/{algorithm_name}_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Interactive Plotly visualization
    if X_pca.shape[1] >= 2:
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels,
            'CustomerID': customer_features['CustomerID'].values[:len(X_pca)] if 'CustomerID' in customer_features.columns else range(len(X_pca))
        })
        
        if X_pca.shape[1] >= 3:
            plot_df['PC3'] = X_pca[:, 2]
            
            fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3',
                               color='Cluster', hover_data=['CustomerID'],
                               title=f'Customer Segments - {algorithm_name}',
                               color_continuous_scale='viridis')
        else:
            fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                            hover_data=['CustomerID'],
                            title=f'Customer Segments - {algorithm_name}')
        
        fig.write_html(f'results/cluster_plots/{algorithm_name}_interactive.html')
    
    print(f"✓ Visualizations saved to results/cluster_plots/")

def analyze_cluster_profiles(customer_features, labels, algorithm_name):
    """Analyze and profile each cluster"""
    print(f"Analyzing cluster profiles for {algorithm_name}...")
    
    # Add cluster labels to customer data
    profile_data = customer_features.copy()
    profile_data['Cluster'] = labels[:len(profile_data)]
    
    # Calculate cluster statistics
    cluster_profiles = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_data = profile_data[profile_data['Cluster'] == cluster_id]
        
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(profile_data) * 100,
            'demographics': {},
            'behavior': {},
            'spending': {},
            'rfm': {}
        }
        
        # Demographic features
        if 'Country' in cluster_data.columns:
            profile['demographics']['top_country'] = cluster_data['Country'].mode().iloc[0] if not cluster_data['Country'].mode().empty else 'Unknown'
        
        # Behavioral features
        behavioral_features = ['MonthlyFrequency', 'AvgDaysBetweenPurchases', 
                              'ProductVarietyRatio', 'CustomerLifetimeMonths']
        
        for feature in behavioral_features:
            if feature in cluster_data.columns:
                profile['behavior'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'std': cluster_data[feature].std(),
                    'median': cluster_data[feature].median()
                }
        
        # Spending features
        spending_features = ['Monetary', 'AvgTransactionValue', 'ValuePerTransaction']
        
        for feature in spending_features:
            if feature in cluster_data.columns:
                profile['spending'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'total': cluster_data[feature].sum() if feature == 'Monetary' else None
                }
        
        # RFM features
        rfm_features = ['Recency', 'Frequency', 'Monetary', 'RFM_Score']
        
        for feature in rfm_features:
            if feature in cluster_data.columns:
                profile['rfm'][feature] = cluster_data[feature].mean()
        
        cluster_profiles[cluster_id] = profile
    
    # Save profiles
    save_results(cluster_profiles, f'results/metrics/{algorithm_name}_profiles.json')
    
    # Create summary table
    summary_data = []
    for cluster_id, profile in cluster_profiles.items():
        summary_data.append({
            'Cluster': cluster_id,
            'Size': profile['size'],
            'Percentage': f"{profile['percentage']:.1f}%",
            'Avg Monetary': f"£{profile['spending'].get('Monetary', {}).get('mean', 0):,.0f}",
            'Avg Frequency': f"{profile['rfm'].get('Frequency', 0):.1f}",
            'Avg Recency': f"{profile['rfm'].get('Recency', 0):.0f} days",
            'RFM Score': f"{profile['rfm'].get('RFM_Score', 0):.1f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'results/metrics/{algorithm_name}_summary.csv', index=False)
    
    print(f"✓ Cluster profiles saved to results/metrics/")
    
    return cluster_profiles, summary_df

def generate_business_insights(cluster_profiles):
    """Generate business insights from cluster profiles"""
    insights = {
        'revenue_generation': {},
        'customer_retention': {},
        'marketing_recommendations': {},
        'segment_definitions': {}
    }
    
    # Find highest revenue cluster
    revenue_by_cluster = {}
    for cluster_id, profile in cluster_profiles.items():
        if 'Monetary' in profile['spending']:
            revenue = profile['spending']['Monetary']['mean'] * profile['size']
            revenue_by_cluster[cluster_id] = revenue
    
    if revenue_by_cluster:
        max_revenue_cluster = max(revenue_by_cluster, key=revenue_by_cluster.get)
        insights['revenue_generation']['highest_value_segment'] = {
            'cluster': int(max_revenue_cluster),
            'estimated_revenue': f"£{revenue_by_cluster[max_revenue_cluster]:,.0f}",
            'percentage_of_total': f"{(revenue_by_cluster[max_revenue_cluster] / sum(revenue_by_cluster.values()) * 100):.1f}%"
        }
    
    # Identify at-risk customers (high recency, low frequency)
    at_risk_clusters = []
    for cluster_id, profile in cluster_profiles.items():
        recency = profile['rfm'].get('Recency', 0)
        frequency = profile['rfm'].get('Frequency', 0)
        monetary = profile['rfm'].get('Monetary', {}).get('mean', 0)
        
        if recency > 180 and frequency < 2:  # Not purchased in 6+ months, few purchases
            at_risk_clusters.append({
                'cluster': cluster_id,
                'recency_days': recency,
                'frequency': frequency,
                'monetary_value': monetary
            })
    
    insights['customer_retention']['at_risk_segments'] = at_risk_clusters
    
    # Define customer segments
    for cluster_id, profile in cluster_profiles.items():
        recency = profile['rfm'].get('Recency', 0)
        frequency = profile['rfm'].get('Frequency', 0)
        monetary = profile['rfm'].get('Monetary', {}).get('mean', 0)
        
        if monetary > 1000 and frequency > 10:
            segment_type = "High-Value Loyalists"
            strategy = "Premium offers, loyalty rewards, exclusive access"
        elif monetary > 500 and frequency > 5:
            segment_type = "Regular Spenders"
            strategy = "Volume discounts, subscription models"
        elif recency > 90 and monetary > 0:
            segment_type = "At-Risk Customers"
            strategy = "Re-engagement campaigns, win-back offers"
        elif frequency > 3 and monetary < 100:
            segment_type = "Budget Shoppers"
            strategy = "Value bundles, budget-friendly promotions"
        else:
            segment_type = "Occasional Buyers"
            strategy = "Awareness campaigns, introductory offers"
        
        insights['segment_definitions'][f"Cluster_{cluster_id}"] = {
            'type': segment_type,
            'characteristics': {
                'size': profile['size'],
                'avg_spend': monetary,
                'purchase_frequency': frequency,
                'recency_days': recency
            },
            'recommended_strategy': strategy
        }
    
    # Marketing recommendations
    insights['marketing_recommendations'] = {
        'premium_targeting': insights['revenue_generation'].get('highest_value_segment', {}),
        'retention_focus': at_risk_clusters[:3] if at_risk_clusters else [],
        'growth_opportunities': [
            cluster_id for cluster_id, profile in cluster_profiles.items()
            if profile['rfm'].get('Frequency', 0) > 5 and profile['rfm'].get('Monetary', {}).get('mean', 0) < 500
        ]
    }
    
    return insights