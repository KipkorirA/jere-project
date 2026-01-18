#!/usr/bin/env python3
"""
Real Estate Price Anomaly Detection System
Phase 6: Anomaly Detection & Pricing Intelligence

This script implements anomaly detection for identifying mispriced properties
and provides pricing intelligence recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PriceAnomalyDetector:
    """
    Class for detecting pricing anomalies and providing market intelligence
    """
    
    def __init__(self, data_path, model_path, scaler_path):
        """Initialize with cleaned data and trained model"""
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        self.df = None
        self.df_with_predictions = None
        self.anomalies_df = None
        self.model = None
        self.scaler = None
        
    def load_data_and_model(self):
        """Load data and trained model"""
        print("="*60)
        print("PHASE 6: PRICE ANOMALY DETECTION")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset: {self.df.shape}")
        
        # Load trained model and scaler
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        print("✅ Model and data loaded successfully!")
        print(f"Model type: {type(self.model).__name__}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for prediction"""
        print("\nPreparing features for anomaly detection...")
        
        # Select same features used in training
        feature_columns = [
            'bedrooms', 'bathrooms', 'toilets', 'furnished', 'serviced', 
            'shared', 'parking', 'is_for_sale', 'price_per_bedroom', 
            'price_per_bathroom', 'total_rooms', 'desirability_score',
            'type_encoded', 'locality_encoded', 'state_encoded',
            'locality_avg_price', 'locality_median_price', 'locality_price_std'
        ]
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        X = X.fillna(X.median())
        
        return X
    
    def predict_prices(self):
        """Generate price predictions for all properties"""
        print("\nGenerating price predictions...")
        
        # Prepare features
        X = self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        log_predictions = self.model.predict(X_scaled)
        price_predictions = np.expm1(log_predictions)  # Convert from log scale
        
        # Add predictions to dataframe
        self.df_with_predictions = self.df.copy()
        self.df_with_predictions['predicted_price'] = price_predictions
        self.df_with_predictions['price_difference'] = self.df_with_predictions['price'] - price_predictions
        self.df_with_predictions['price_ratio'] = self.df_with_predictions['price'] / price_predictions
        
        print(f"✅ Generated {len(price_predictions):,} price predictions")
        
        return self.df_with_predictions
    
    def detect_anomalies(self):
        """Detect pricing anomalies using multiple methods"""
        print("\nDetecting pricing anomalies...")
        
        # Method 1: Percentage-based anomaly detection
        # Calculate percentage difference from predicted price
        percentage_diff = abs(self.df_with_predictions['price_difference']) / self.df_with_predictions['predicted_price']
        
        # Method 2: Z-score based detection
        price_z_scores = np.abs((self.df_with_predictions['price'] - self.df_with_predictions['price'].mean()) / 
                              self.df_with_predictions['price'].std())
        
        # Method 3: IQR-based detection for residuals
        residual_q1 = self.df_with_predictions['price_difference'].quantile(0.25)
        residual_q3 = self.df_with_predictions['price_difference'].quantile(0.75)
        residual_iqr = residual_q3 - residual_q1
        
        # Define anomaly thresholds
        percentage_threshold = 0.30  # 30% deviation
        z_score_threshold = 2.0  # 2 standard deviations
        iqr_multiplier = 1.5
        
        # Create anomaly flags
        self.df_with_predictions['percentage_anomaly'] = percentage_diff > percentage_threshold
        self.df_with_predictions['z_score_anomaly'] = price_z_scores > z_score_threshold
        self.df_with_predictions['residual_anomaly'] = (
            (self.df_with_predictions['price_difference'] < (residual_q1 - iqr_multiplier * residual_iqr)) |
            (self.df_with_predictions['price_difference'] > (residual_q3 + iqr_multiplier * residual_iqr))
        )
        
        # Overall anomaly flag (any method detects it)
        self.df_with_predictions['is_anomaly'] = (
            self.df_with_predictions['percentage_anomaly'] |
            self.df_with_predictions['z_score_anomaly'] |
            self.df_with_predictions['residual_anomaly']
        )
        
        # Classify anomalies as underpriced, overpriced, or fairly priced
        def classify_price_status(row):
            if not row['is_anomaly']:
                return 'Fairly Priced'
            elif row['price_difference'] < 0:
                return 'Underpriced'
            else:
                return 'Overpriced'
        
        self.df_with_predictions['price_status'] = self.df_with_predictions.apply(classify_price_status, axis=1)
        
        # Calculate anomaly statistics
        total_properties = len(self.df_with_predictions)
        anomalies = self.df_with_predictions['is_anomaly'].sum()
        underpriced = (self.df_with_predictions['price_status'] == 'Underpriced').sum()
        overpriced = (self.df_with_predictions['price_status'] == 'Overpriced').sum()
        fairly_priced = (self.df_with_predictions['price_status'] == 'Fairly Priced').sum()
        
        print(f"\n📊 ANOMALY DETECTION RESULTS:")
        print(f"Total properties analyzed: {total_properties:,}")
        print(f"Anomalies detected: {anomalies:,} ({anomalies/total_properties*100:.1f}%)")
        print(f"Underpriced properties: {underpriced:,} ({underpriced/total_properties*100:.1f}%)")
        print(f"Overpriced properties: {overpriced:,} ({overpriced/total_properties*100:.1f}%)")
        print(f"Fairly priced properties: {fairly_priced:,} ({fairly_priced/total_properties*100:.1f}%)")
        
        return self.df_with_predictions
    
    def analyze_anomalies(self):
        """Analyze patterns in detected anomalies"""
        print("\n" + "="*50)
        print("ANOMALY ANALYSIS")
        print("="*50)
        
        # Filter anomalies
        anomalies = self.df_with_predictions[self.df_with_predictions['is_anomaly']].copy()
        
        if len(anomalies) == 0:
            print("No anomalies detected.")
            return
        
        print(f"\n📈 Top 10 Most Underpriced Properties:")
        underpriced = anomalies[anomalies['price_status'] == 'Underpriced'].nsmallest(10, 'price_difference')
        for idx, row in underpriced.iterrows():
            print(f"ID {row['id']}: {row['type']} in {row['locality']}")
            print(f"  Actual: ${row['price']:,.0f} | Predicted: ${row['predicted_price']:,.0f}")
            print(f"  Underpriced by: ${abs(row['price_difference']):,.0f} ({abs(row['price_difference'])/row['predicted_price']*100:.1f}%)")
            print()
        
        print(f"\n📉 Top 10 Most Overpriced Properties:")
        overpriced = anomalies[anomalies['price_status'] == 'Overpriced'].nlargest(10, 'price_difference')
        for idx, row in overpriced.iterrows():
            print(f"ID {row['id']}: {row['type']} in {row['locality']}")
            print(f"  Actual: ${row['price']:,.0f} | Predicted: ${row['predicted_price']:,.0f}")
            print(f"  Overpriced by: ${row['price_difference']:,.0f} ({row['price_difference']/row['predicted_price']*100:.1f}%)")
            print()
        
        # Anomaly patterns by location
        print(f"\n🏙️ Anomaly Distribution by Location (Top 10):")
        location_anomalies = anomalies.groupby('locality').agg({
            'is_anomaly': 'count',
            'price_status': lambda x: (x == 'Overpriced').sum()
        }).rename(columns={'is_anomaly': 'total_anomalies', 'price_status': 'overpriced_count'})
        location_anomalies['anomaly_rate'] = location_anomalies['total_anomalies'] / location_anomalies.groupby('locality').transform('size') * 100
        location_anomalies = location_anomalies.sort_values('total_anomalies', ascending=False).head(10)
        print(location_anomalies.round(1))
        
        # Anomaly patterns by property type
        print(f"\n🏠 Anomaly Distribution by Property Type:")
        type_anomalies = anomalies.groupby('type').agg({
            'is_anomaly': 'count',
            'price_status': lambda x: pd.Series([(x == 'Overpriced').sum(), (x == 'Underpriced').sum()])
        })
        print(type_anomalies)
        
        return anomalies
    
    def create_anomaly_visualizations(self):
        """Create comprehensive anomaly detection visualizations"""
        print("\nCreating anomaly detection visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # 1. Price vs Predicted Price Scatter Plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.df_with_predictions['predicted_price'], 
                             self.df_with_predictions['price'], 
                             c=self.df_with_predictions['is_anomaly'], 
                             cmap='RdYlBu', alpha=0.6)
        ax1.plot([self.df_with_predictions['price'].min(), self.df_with_predictions['price'].max()], 
                  [self.df_with_predictions['price'].min(), self.df_with_predictions['price'].max()], 
                  'r--', lw=2)
        ax1.set_xlabel('Predicted Price ($)')
        ax1.set_ylabel('Actual Price ($)')
        ax1.set_title('Actual vs Predicted Prices (Anomalies in Red/Blue)')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. Price Status Distribution
        ax2 = axes[0, 1]
        status_counts = self.df_with_predictions['price_status'].value_counts()
        colors = ['green', 'red', 'orange']
        status_counts.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title('Price Status Distribution')
        ax2.set_xlabel('Price Status')
        ax2.set_ylabel('Number of Properties')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Price Difference Distribution
        ax3 = axes[1, 0]
        self.df_with_predictions['price_difference'].hist(bins=50, alpha=0.7, ax=ax3)
        ax3.axvline(0, color='red', linestyle='--', lw=2)
        ax3.set_title('Price Difference Distribution')
        ax3.set_xlabel('Price Difference (Actual - Predicted)')
        ax3.set_ylabel('Frequency')
        
        # 4. Anomaly Rate by Location (Top 15)
        ax4 = axes[1, 1]
        location_stats = self.df_with_predictions.groupby('locality').agg({
            'is_anomaly': 'mean',
            'id': 'count'
        }).rename(columns={'is_anomaly': 'anomaly_rate', 'id': 'property_count'})
        location_stats = location_stats[location_stats['property_count'] >= 10]  # Only locations with 10+ properties
        location_stats = location_stats.sort_values('anomaly_rate', ascending=False).head(15)
        
        location_stats['anomaly_rate'].plot(kind='barh', ax=ax4)
        ax4.set_title('Anomaly Rate by Location (Top 15, Min 10 Properties)')
        ax4.set_xlabel('Anomaly Rate')
        
        # 5. Price Ratio Distribution
        ax5 = axes[2, 0]
        price_ratios = self.df_with_predictions['price_ratio']
        # Filter extreme outliers for better visualization
        filtered_ratios = price_ratios[(price_ratios > 0.1) & (price_ratios < 10)]
        filtered_ratios.hist(bins=50, alpha=0.7, ax=ax5)
        ax5.axvline(1, color='red', linestyle='--', lw=2)
        ax5.set_title('Price Ratio Distribution (Actual/Predicted)')
        ax5.set_xlabel('Price Ratio')
        ax5.set_ylabel('Frequency')
        ax5.set_xlim(0.1, 10)
        
        # 6. Feature Importance for Anomaly Detection
        ax6 = axes[2, 1]
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            feature_names = self.prepare_features().columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            importance_df.plot(x='feature', y='importance', kind='barh', ax=ax6)
            ax6.set_title('Feature Importance in Price Prediction')
            ax6.set_xlabel('Importance')
        else:
            ax6.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig('/home/aron/code/project/anomaly_detection_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        anomalies = self.df_with_predictions[self.df_with_predictions['is_anomaly']]
        
        # Market transparency insights
        total_props = len(self.df_with_predictions)
        anomaly_rate = (len(anomalies) / total_props) * 100
        
        print(f"\n🎯 MARKET TRANSPARENCY INSIGHTS:")
        print(f"• {100-anomaly_rate:.1f}% of properties are fairly priced")
        print(f"• {anomaly_rate:.1f}% show significant price deviations")
        
        if anomaly_rate > 20:
            print("• HIGH anomaly rate suggests potential market inefficiencies")
        elif anomaly_rate > 10:
            print("• MODERATE anomaly rate indicates relatively stable pricing")
        else:
            print("• LOW anomaly rate suggests efficient market pricing")
        
        # Buyer insights
        underpriced = anomalies[anomalies['price_status'] == 'Underpriced']
        if len(underpriced) > 0:
            print(f"\n🏠 BUYER OPPORTUNITIES:")
            print(f"• {len(underpriced):,} underpriced properties identified")
            avg_underprice = underpriced['price_difference'].mean()
            print(f"• Average potential savings: ${abs(avg_underprice):,.0f}")
            
            # Best value locations
            best_locations = underpriced.groupby('locality').agg({
                'price_difference': 'mean',
                'id': 'count'
            }).rename(columns={'price_difference': 'avg_underprice', 'id': 'count'})
            best_locations = best_locations[best_locations['count'] >= 3].sort_values('avg_underprice').head(5)
            
            print(f"• Top undervalued locations:")
            for locality, data in best_locations.iterrows():
                print(f"  - {locality}: {data['count']} properties, avg savings ${abs(data['avg_underprice']):,.0f}")
        
        # Seller insights
        overpriced = anomalies[anomalies['price_status'] == 'Overpriced']
        if len(overpriced) > 0:
            print(f"\n💰 SELLER INSIGHTS:")
            print(f"• {len(overpriced):,} overpriced properties identified")
            avg_overprice = overpriced['price_difference'].mean()
            print(f"• Average overpricing: ${avg_overprice:,.0f}")
            
            # Locations with most overpriced properties
            overpriced_locations = overpriced.groupby('locality')['id'].count().sort_values(ascending=False).head(5)
            print(f"• Markets with highest overpricing:")
            for locality, count in overpriced_locations.items():
                print(f"  - {locality}: {count} overpriced properties")
        
        # Investment insights
        print(f"\n📊 INVESTMENT RECOMMENDATIONS:")
        
        # Properties with high price-to-prediction ratios (potentially undervalued)
        potential_deals = self.df_with_predictions[
            (self.df_with_predictions['price_ratio'] < 0.8) & 
            (self.df_with_predictions['price'] < self.df_with_predictions['predicted_price'] * 0.8)
        ].sort_values('price_ratio')
        
        if len(potential_deals) > 0:
            print(f"• {len(potential_deals)} potential investment opportunities identified")
            print(f"• Best potential ROI locations:")
            investment_locations = potential_deals.groupby('locality').agg({
                'price_ratio': 'mean',
                'id': 'count'
            }).rename(columns={'price_ratio': 'avg_ratio', 'id': 'count'})
            investment_locations = investment_locations[investment_locations['count'] >= 2].sort_values('avg_ratio').head(5)
            for locality, data in investment_locations.iterrows():
                print(f"  - {locality}: avg {data['avg_ratio']:.2f}x price ratio, {data['count']} opportunities")
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_rate': anomaly_rate,
            'underpriced_count': len(underpriced) if len(underpriced) > 0 else 0,
            'overpriced_count': len(overpriced) if len(overpriced) > 0 else 0,
            'potential_deals': len(potential_deals) if len(potential_deals) > 0 else 0
        }
    
    def save_results(self):
        """Save anomaly detection results"""
        print("\nSaving results...")
        
        # Save complete dataset with predictions and anomaly flags
        self.df_with_predictions.to_csv('/home/aron/code/project/properties_with_anomaly_analysis.csv', index=False)
        
        # Save just the anomalies for focused analysis
        anomalies_only = self.df_with_predictions[self.df_with_predictions['is_anomaly']]
        anomalies_only.to_csv('/home/aron/code/project/detected_price_anomalies.csv', index=False)
        
        # Create summary report
        summary_stats = {
            'total_properties': len(self.df_with_predictions),
            'anomalies_detected': self.df_with_predictions['is_anomaly'].sum(),
            'underpriced': (self.df_with_predictions['price_status'] == 'Underpriced').sum(),
            'overpriced': (self.df_with_predictions['price_status'] == 'Overpriced').sum(),
            'fairly_priced': (self.df_with_predictions['price_status'] == 'Fairly Priced').sum(),
            'average_prediction_accuracy': 1 - (abs(self.df_with_predictions['price_difference']).mean() / self.df_with_predictions['price'].mean())
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv('/home/aron/code/project/anomaly_detection_summary.csv', index=False)
        
        print("✅ Results saved:")
        print("• properties_with_anomaly_analysis.csv - Complete dataset with predictions")
        print("• detected_price_anomalies.csv - Anomalous properties only")
        print("• anomaly_detection_summary.csv - Summary statistics")
        print("• anomaly_detection_analysis.png - Visualization charts")
        
        return summary_stats
    
    def run_complete_pipeline(self):
        """Run the complete anomaly detection pipeline"""
        try:
            # Load data and model
            self.load_data_and_model()
            
            # Generate predictions
            self.predict_prices()
            
            # Detect anomalies
            self.detect_anomalies()
            
            # Analyze anomalies
            anomalies = self.analyze_anomalies()
            
            # Create visualizations
            self.create_anomaly_visualizations()
            
            # Generate business insights
            insights = self.generate_business_insights()
            
            # Save results
            summary_stats = self.save_results()
            
            print("\n🎉 PHASE 6 COMPLETED SUCCESSFULLY!")
            print("Price anomaly detection system is ready!")
            
            return insights, summary_stats
            
        except Exception as e:
            print(f"❌ Error in anomaly detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function"""
    print("🔍 Starting Price Anomaly Detection System")
    print("="*60)
    
    # Initialize detector
    detector = PriceAnomalyDetector(
        data_path='/home/aron/code/project/cleaned_real_estate_data.csv',
        model_path='/home/aron/code/project/best_price_model.pkl',
        scaler_path='/home/aron/code/project/feature_scaler.pkl'
    )
    
    # Run complete pipeline
    insights, summary_stats = detector.run_complete_pipeline()
    
    return detector, insights, summary_stats

if __name__ == "__main__":
    detector, insights, summary_stats = main()
