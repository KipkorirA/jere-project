#!/usr/bin/env python3
"""
Real Estate Pricing Intelligence System
Kenyan Real Estate Market Analysis

This script implements a comprehensive machine learning-based pricing intelligence system
for predicting property values and identifying pricing anomalies in the Kenyan real estate market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RealEstateAnalyzer:
    """
    Main class for real estate price analysis and anomaly detection
    """
    
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """Load dataset and perform initial exploration"""
        print("="*60)
        print("PHASE 1: DATA LOADING & EXPLORATION")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nBasic statistics:")
        print(self.df.describe())
        
        return self.df
    
    def assess_data_quality(self):
        """Comprehensive data quality assessment"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Missing values
        print("\nMissing Values Analysis:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
        print(missing_df)
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Data type issues
        print("\nData Type Analysis:")
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            print(f"{col}: {unique_vals} unique values, Type: {self.df[col].dtype}")
            if self.df[col].dtype == 'object':
                print(f"  Sample values: {self.df[col].dropna().unique()[:5]}")
        
        # Price analysis
        print("\nPrice Column Analysis:")
        print(f"Price column stats:")
        print(self.df['price'].describe())
        
        # Category analysis
        print("\nProperty Category Distribution:")
        print(self.df['category'].value_counts())
        
        # Location analysis
        print("\nTop 10 Localities:")
        print(self.df['locality'].value_counts().head(10))
        
        return missing_df
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        print("\n" + "="*60)
        print("PHASE 2: DATA CLEANING")
        print("="*60)
        
        self.df_clean = self.df.copy()
        
        # Remove duplicates
        initial_count = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates()
        print(f"Removed {initial_count - len(self.df_clean)} duplicate rows")
        
        # Handle missing values strategically
        print("\nHandling missing values...")
        
        # For numerical columns, fill with median
        numerical_cols = ['bedrooms', 'bathrooms', 'toilets', 'parking']
        for col in numerical_cols:
            if col in self.df_clean.columns:
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
                self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = ['furnished', 'serviced', 'shared']
        for col in categorical_cols:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean[col].fillna(self.df_clean[col].mode()[0] if not self.df_clean[col].mode().empty else 0)
        
        # Clean price column - remove non-numeric entries
        print("Cleaning price data...")
        self.df_clean['price'] = pd.to_numeric(self.df_clean['price'], errors='coerce')
        
        # Remove rows with invalid prices (negative, zero, or extremely high)
        price_before = len(self.df_clean)
        self.df_clean = self.df_clean[
            (self.df_clean['price'] > 0) & 
            (self.df_clean['price'] < 1e9)  # Remove unrealistic prices
        ]
        print(f"Removed {price_before - len(self.df_clean)} rows with invalid prices")
        
        # Create price categories based on property type
        self.df_clean['is_for_sale'] = (self.df_clean['category'] == 'For Sale').astype(int)
        
        # Focus on residential properties initially
        residential_types = ['House', 'Apartment']
        self.df_clean = self.df_clean[self.df_clean['type'].isin(residential_types)]
        print(f"Filtered to residential properties: {len(self.df_clean)} rows")
        
        # Handle outliers using IQR method for price
        Q1 = self.df_clean['price'].quantile(0.25)
        Q3 = self.df_clean['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(self.df_clean)
        self.df_clean = self.df_clean[
            (self.df_clean['price'] >= lower_bound) & 
            (self.df_clean['price'] <= upper_bound)
        ]
        print(f"Removed {outliers_before - len(self.df_clean)} price outliers")
        
        print(f"\nCleaned dataset shape: {self.df_clean.shape}")
        print("Data cleaning completed!")
        
        return self.df_clean
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*60)
        print("PHASE 3: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Set up the plotting environment
        plt.figure(figsize=(20, 15))
        
        # 1. Price distribution analysis
        plt.subplot(3, 4, 1)
        self.df_clean['price'].hist(bins=50, alpha=0.7)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # 2. Log price distribution (for better visualization)
        plt.subplot(3, 4, 2)
        log_prices = np.log1p(self.df_clean['price'])
        log_prices.hist(bins=50, alpha=0.7)
        plt.title('Log Price Distribution')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        
        # 3. Price by property type
        plt.subplot(3, 4, 3)
        self.df_clean.boxplot(column='price', by='type', ax=plt.gca())
        plt.title('Price by Property Type')
        plt.suptitle('')
        
        # 4. Price by category (Sale vs Rent)
        plt.subplot(3, 4, 4)
        self.df_clean.boxplot(column='price', by='category', ax=plt.gca())
        plt.title('Price by Category')
        plt.suptitle('')
        
        # 5. Bedrooms vs Price
        plt.subplot(3, 4, 5)
        plt.scatter(self.df_clean['bedrooms'], self.df_clean['price'], alpha=0.5)
        plt.xlabel('Bedrooms')
        plt.ylabel('Price')
        plt.title('Bedrooms vs Price')
        
        # 6. Bathrooms vs Price
        plt.subplot(3, 4, 6)
        plt.scatter(self.df_clean['bathrooms'], self.df_clean['price'], alpha=0.5)
        plt.xlabel('Bathrooms')
        plt.ylabel('Price')
        plt.title('Bathrooms vs Price')
        
        # 7. Top localities by average price
        plt.subplot(3, 4, 7)
        top_localities = self.df_clean.groupby('locality')['price'].mean().sort_values(ascending=False).head(10)
        top_localities.plot(kind='barh')
        plt.title('Top 10 Localities by Avg Price')
        plt.xlabel('Average Price')
        
        # 8. Furnished vs Unfurnished
        plt.subplot(3, 4, 8)
        furnished_data = self.df_clean.groupby('furnished')['price'].mean()
        furnished_data.plot(kind='bar')
        plt.title('Furnished vs Unfurnished Prices')
        plt.xlabel('Furnished (0=No, 1=Yes)')
        plt.ylabel('Average Price')
        plt.xticks(rotation=0)
        
        # 9. Correlation heatmap
        plt.subplot(3, 4, 9)
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df_clean[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 10. Price distribution by bedroom count
        plt.subplot(3, 4, 10)
        for bedrooms in sorted(self.df_clean['bedrooms'].unique())[:6]:  # Top 6 bedroom counts
            subset = self.df_clean[self.df_clean['bedrooms'] == bedrooms]
            if len(subset) > 10:  # Only plot if sufficient data
                subset['price'].hist(alpha=0.5, label=f'{bedrooms} BR', bins=30)
        plt.title('Price Distribution by Bedroom Count')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 11. Price trends by locality (top 5)
        plt.subplot(3, 4, 11)
        top_5_localities = self.df_clean['locality'].value_counts().head(5).index
        for locality in top_5_localities:
            subset = self.df_clean[self.df_clean['locality'] == locality]
            plt.scatter(subset['bedrooms'], subset['price'], alpha=0.6, label=locality)
        plt.xlabel('Bedrooms')
        plt.ylabel('Price')
        plt.title('Price vs Bedrooms by Top Localities')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 12. Parking impact on price
        plt.subplot(3, 4, 12)
        parking_impact = self.df_clean.groupby('parking')['price'].mean()
        parking_impact.plot(kind='bar')
        plt.title('Parking Impact on Price')
        plt.xlabel('Parking Spaces')
        plt.ylabel('Average Price')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('/home/aron/code/project/eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical summary
        print("\nKey Statistics:")
        print(f"Total properties analyzed: {len(self.df_clean)}")
        print(f"Price range: ${self.df_clean['price'].min():,.0f} - ${self.df_clean['price'].max():,.0f}")
        print(f"Median price: ${self.df_clean['price'].median():,.0f}")
        print(f"Average price: ${self.df_clean['price'].mean():,.0f}")
        
        print(f"\nProperty type distribution:")
        print(self.df_clean['type'].value_counts())
        
        print(f"\nTop 5 localities by count:")
        print(self.df_clean['locality'].value_counts().head())
        
        return self.df_clean
    
    def feature_engineering(self):
        """Create additional features for modeling"""
        print("\n" + "="*60)
        print("PHASE 4: FEATURE ENGINEERING")
        print("="*60)
        
        df_features = self.df_clean.copy()
        
        # Price-derived features
        df_features['price_per_bedroom'] = df_features['price'] / (df_features['bedrooms'] + 1)  # +1 to avoid division by zero
        df_features['price_per_bathroom'] = df_features['price'] / (df_features['bathrooms'] + 1)
        df_features['total_rooms'] = df_features['bedrooms'] + df_features['bathrooms']
        
        # Price categories
        price_terciles = df_features['price'].quantile([0.33, 0.67])
        df_features['price_category'] = pd.cut(
            df_features['price'], 
            bins=[0, price_terciles.iloc[0], price_terciles.iloc[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        # Location-based features
        locality_stats = df_features.groupby('locality').agg({
            'price': ['mean', 'median', 'std']
        }).round(2)
        locality_stats.columns = ['locality_avg_price', 'locality_median_price', 'locality_price_std']
        locality_stats = locality_stats.reset_index()
        
        df_features = df_features.merge(locality_stats, on='locality', how='left')
        
        # Property desirability score (simple heuristic)
        df_features['desirability_score'] = (
            df_features['bedrooms'] * 0.3 +
            df_features['bathrooms'] * 0.2 +
            df_features['parking'] * 0.2 +
            df_features['furnished'] * 0.15 +
            df_features['serviced'] * 0.15
        )
        
        # Encode categorical variables
        le_type = LabelEncoder()
        le_locality = LabelEncoder()
        le_state = LabelEncoder()
        
        df_features['type_encoded'] = le_type.fit_transform(df_features['type'])
        df_features['locality_encoded'] = le_locality.fit_transform(df_features['locality'])
        df_features['state_encoded'] = le_state.fit_transform(df_features['state'])
        
        # Log transform price for modeling (reduces skewness)
        df_features['log_price'] = np.log1p(df_features['price'])
        
        print("Feature engineering completed!")
        print(f"New features created: {df_features.shape[1] - self.df_clean.shape[1]}")
        print(f"Total features available: {df_features.shape[1]}")
        
        # Store encoders for later use
        self.encoders = {
            'type': le_type,
            'locality': le_locality,
            'state': le_state
        }
        
        return df_features

def main():
    """Main execution function"""
    print("🏠 Real Estate Pricing Intelligence System")
    print("📊 Kenyan Real Estate Market Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RealEstateAnalyzer('/home/aron/code/project/kenya_listings.csv')
    
    # Execute analysis pipeline
    try:
        # Phase 1: Data Loading & Exploration
        df = analyzer.load_and_explore_data()
        
        # Phase 1 Continued: Data Quality Assessment
        missing_analysis = analyzer.assess_data_quality()
        
        # Phase 2: Data Cleaning
        df_clean = analyzer.clean_data()
        
        # Phase 3: Exploratory Data Analysis
        df_eda = analyzer.exploratory_data_analysis()
        
        # Phase 4: Feature Engineering
        df_features = analyzer.feature_engineering()
        
        # Save cleaned dataset
        df_features.to_csv('/home/aron/code/project/cleaned_real_estate_data.csv', index=False)
        print(f"\n✅ Cleaned dataset saved as 'cleaned_real_estate_data.csv'")
        print(f"📈 Analysis charts saved as 'eda_analysis.png'")
        
        # Save analyzer for later use
        analyzer.df_features = df_features
        analyzer.df_clean = df_clean
        
        print("\n🎉 Phase 1-4 completed successfully!")
        print("Ready for modeling phase...")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyzer = main()
