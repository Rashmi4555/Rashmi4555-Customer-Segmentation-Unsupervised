"""
Feature engineering module for customer segmentation
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Creates customer-level features from transaction data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_customer_features(self, df):
        """Create customer-level features from transaction data"""
        print("\nCreating customer features...")
        
        # Set reference date (day after last purchase)
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
        # Group by customer
        customer_data = df.groupby('CustomerID').agg({
            'InvoiceDate': ['max', 'nunique'],
            'TotalValue': ['sum', 'mean'],
            'Quantity': ['sum', 'mean'],
            'StockCode': 'nunique',
            'InvoiceNo': 'nunique',
            'Country': 'first'
        }).reset_index()
        
        # Flatten column names
        customer_data.columns = ['CustomerID', 'LastPurchaseDate', 'UniqueDates',
                                'TotalSpent', 'AvgTransactionValue', 'TotalQuantity',
                                'AvgQuantity', 'UniqueProducts', 'TransactionCount',
                                'Country']
        
        # Calculate RFM metrics
        customer_data = self._calculate_rfm(customer_data, reference_date)
        
        # Calculate behavioral metrics
        customer_data = self._calculate_behavioral_metrics(customer_data, df)
        
        # Calculate derived ratios
        customer_data = self._calculate_derived_ratios(customer_data)
        
        # Encode categorical variables
        customer_data = self._encode_categorical(customer_data)
        
        print(f"✓ Created features for {len(customer_data):,} customers")
        print(f"  Features: {len(customer_data.columns)} columns")
        
        return customer_data
    
    def _calculate_rfm(self, customer_data, reference_date):
        """Calculate Recency, Frequency, Monetary metrics"""
        # Recency: Days since last purchase
        customer_data['Recency'] = (reference_date - customer_data['LastPurchaseDate']).dt.days
        
        # Frequency: Number of transactions
        customer_data['Frequency'] = customer_data['TransactionCount']
        
        # Monetary: Total amount spent
        customer_data['Monetary'] = customer_data['TotalSpent']
        
        # RFM Scores (1-5 scale, 5 being best)
        for metric in ['Recency', 'Frequency', 'Monetary']:
            # For Recency, lower is better (more recent)
            if metric == 'Recency':
                customer_data[f'{metric}_Score'] = pd.qcut(
                    customer_data[metric], 
                    5, 
                    labels=[5, 4, 3, 2, 1]
                ).astype(int)
            else:
                customer_data[f'{metric}_Score'] = pd.qcut(
                    customer_data[metric], 
                    5, 
                    labels=[1, 2, 3, 4, 5]
                ).astype(int)
        
        # Combined RFM Score
        customer_data['RFM_Score'] = (
            customer_data['Recency_Score'] + 
            customer_data['Frequency_Score'] + 
            customer_data['Monetary_Score']
        )
        
        # RFM Segment
        customer_data['RFM_Segment'] = pd.cut(
            customer_data['RFM_Score'],
            bins=[0, 5, 8, 11, 15],
            labels=['Low Value', 'Medium Value', 'High Value', 'Top Value']
        )
        
        return customer_data
    
    def _calculate_behavioral_metrics(self, customer_data, df):
        """Calculate behavioral patterns"""
        
        # Calculate purchase frequency (transactions per month)
        customer_first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        customer_first_purchase.columns = ['CustomerID', 'FirstPurchaseDate']
        
        # Merge first purchase date
        customer_data = customer_data.merge(customer_first_purchase, on='CustomerID')
        
        # Calculate customer lifetime in months
        customer_data['CustomerLifetimeMonths'] = (
            (customer_data['LastPurchaseDate'] - customer_data['FirstPurchaseDate']).dt.days / 30
        ).clip(lower=1)  # Minimum 1 month to avoid division by zero
        
        # Monthly frequency
        customer_data['MonthlyFrequency'] = (
            customer_data['Frequency'] / customer_data['CustomerLifetimeMonths']
        )
        
        # Average days between purchases
        if customer_data['Frequency'].max() > 1:
            customer_data['AvgDaysBetweenPurchases'] = (
                customer_data['Recency'] / (customer_data['Frequency'] - 1)
            ).fillna(0)
        else:
            customer_data['AvgDaysBetweenPurchases'] = 0
        
        # Product variety ratio
        customer_data['ProductVarietyRatio'] = (
            customer_data['UniqueProducts'] / customer_data['TotalQuantity']
        )
        
        # Price sensitivity (inverse of average price)
        customer_data['PriceSensitivity'] = (
            customer_data['TotalQuantity'] / customer_data['TotalSpent']
        )
        
        return customer_data
    
    def _calculate_derived_ratios(self, customer_data):
        """Calculate important derived ratios"""
        
        # Value per transaction
        customer_data['ValuePerTransaction'] = (
            customer_data['Monetary'] / customer_data['Frequency']
        )
        
        # Quantity per transaction
        customer_data['QuantityPerTransaction'] = (
            customer_data['TotalQuantity'] / customer_data['Frequency']
        )
        
        # Customer activity ratio (recent purchases / total purchases)
        customer_data['ActivityRatio'] = (
            1 / (customer_data['Recency'] + 1) * customer_data['Frequency']
        )
        
        # Spending consistency (CV of transaction values would require full history)
        # For now, use frequency to monetary ratio
        customer_data['SpendingConsistency'] = (
            customer_data['Frequency'] / customer_data['Monetary']
        )
        
        # Return all numeric columns for clustering
        numeric_features = [
            'Recency', 'Frequency', 'Monetary',
            'MonthlyFrequency', 'AvgDaysBetweenPurchases',
            'ProductVarietyRatio', 'PriceSensitivity',
            'ValuePerTransaction', 'QuantityPerTransaction',
            'ActivityRatio', 'CustomerLifetimeMonths'
        ]
        
        # Keep only columns that exist
        numeric_features = [col for col in numeric_features if col in customer_data.columns]
        
        # Add RFM scores
        numeric_features.extend(['Recency_Score', 'Frequency_Score', 'Monetary_Score', 'RFM_Score'])
        
        # Store feature names
        self.feature_names = numeric_features
        
        return customer_data
    
    def _encode_categorical(self, customer_data):
        """Encode categorical variables"""
        if 'Country' in customer_data.columns:
            le = LabelEncoder()
            customer_data['Country_Encoded'] = le.fit_transform(customer_data['Country'])
            self.label_encoders['Country'] = le
        
        if 'RFM_Segment' in customer_data.columns:
            le = LabelEncoder()
            customer_data['RFM_Segment_Encoded'] = le.fit_transform(customer_data['RFM_Segment'])
            self.label_encoders['RFM_Segment'] = le
        
        return customer_data
    
    def scale_features(self, customer_data):
        """Scale features for clustering"""
        print("Scaling features...")
        
        # Get numeric features
        if not hasattr(self, 'feature_names'):
            self.feature_names = customer_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove CustomerID from features
        self.feature_names = [col for col in self.feature_names if col not in ['CustomerID']]
        
        # Scale features
        X = customer_data[self.feature_names].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✓ Scaled {X_scaled.shape[1]} features")
        
        return X_scaled, self.scaler
    
    def apply_pca(self, X_scaled, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        print(f"Applying PCA (variance: {n_components*100:.0f}%)...")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"✓ Reduced to {X_pca.shape[1]} components")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca, pca