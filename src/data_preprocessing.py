"""
Data preprocessing module for customer segmentation
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self):
        self.report_data = {}
    
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        print(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath, encoding='latin1')
            print(f"✓ Loaded {len(df):,} records")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def clean_data(self, df):
        """Apply comprehensive data cleaning"""
        print("Cleaning data...")
        original_shape = df.shape
        
        # Create a copy
        df_clean = df.copy()
        
        # 1. Handle missing values
        missing_customers = df_clean['CustomerID'].isnull().sum()
        df_clean = df_clean.dropna(subset=['CustomerID'])
        print(f"  Removed {missing_customers:,} records with missing CustomerID")
        
        # 2. Remove cancelled transactions
        cancelled_mask = df_clean['InvoiceNo'].astype(str).str.startswith('C')
        cancelled_count = cancelled_mask.sum()
        df_clean = df_clean[~cancelled_mask]
        print(f"  Removed {cancelled_count:,} cancelled transactions")
        
        # 3. Remove invalid quantities and prices
        invalid_qty = (df_clean['Quantity'] <= 0).sum()
        invalid_price = (df_clean['UnitPrice'] <= 0).sum()
        df_clean = df_clean[df_clean['Quantity'] > 0]
        df_clean = df_clean[df_clean['UnitPrice'] > 0]
        print(f"  Removed {invalid_qty:,} invalid quantities")
        print(f"  Removed {invalid_price:,} invalid prices")
        
        # 4. Convert date column
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        
        # 5. Create total value column
        df_clean['TotalValue'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        # 6. Handle outliers using IQR method
        df_clean = self._handle_outliers(df_clean)
        
        # 7. Basic validation
        df_clean = self._validate_data(df_clean)
        
        final_shape = df_clean.shape
        print(f"\n✓ Cleaning complete:")
        print(f"  Original: {original_shape[0]:,} records")
        print(f"  Final: {final_shape[0]:,} records")
        print(f"  Removed: {original_shape[0] - final_shape[0]:,} records ({((original_shape[0] - final_shape[0])/original_shape[0]*100):.1f}%)")
        
        # Store report data
        self.report_data['cleaning_stats'] = {
            'original_records': original_shape[0],
            'final_records': final_shape[0],
            'removed_records': original_shape[0] - final_shape[0],
            'removed_percentage': ((original_shape[0] - final_shape[0])/original_shape[0]*100)
        }
        
        return df_clean
    
    def _handle_outliers(self, df):
        """Cap outliers using IQR method"""
        numeric_cols = ['Quantity', 'UnitPrice', 'TotalValue']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"  Capped {outliers:,} outliers in {col}")
        
        return df
    
    def _validate_data(self, df):
        """Validate cleaned data"""
        # Check for negative values
        negative_qty = (df['Quantity'] < 0).sum()
        negative_price = (df['UnitPrice'] < 0).sum()
        
        if negative_qty > 0 or negative_price > 0:
            print(f"  ⚠️ Found {negative_qty} negative quantities and {negative_price} negative prices")
        
        # Check date range
        date_range = df['InvoiceDate'].max() - df['InvoiceDate'].min()
        print(f"  Time span: {date_range.days} days")
        
        return df
    
    def get_summary(self):
        """Get cleaning summary"""
        return self.report_data