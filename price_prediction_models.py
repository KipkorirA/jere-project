#!/usr/bin/env python3
"""
Real Estate Price Prediction Models
Phase 5: Machine Learning Model Development

This script implements multiple machine learning models for price prediction
and includes comprehensive model evaluation and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PricePredictionModels:
    """
    Class for implementing and comparing multiple price prediction models
    """
    
    def __init__(self, data_path):
        """Initialize with cleaned data path"""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the cleaned dataset"""
        print("="*60)
        print("PHASE 5: MACHINE LEARNING MODELING")
        print("="*60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded cleaned dataset: {self.df.shape}")
        
        # Display key statistics
        print(f"\nDataset Overview:")
        print(f"Properties: {len(self.df):,}")
        print(f"Price range: ${self.df['price'].min():,.0f} - ${self.df['price'].max():,.0f}")
        print(f"Median price: ${self.df['price'].median():,.0f}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\nPreparing features for modeling...")
        
        # Select relevant features for modeling
        feature_columns = [
            'bedrooms', 'bathrooms', 'toilets', 'furnished', 'serviced', 
            'shared', 'parking', 'is_for_sale', 'price_per_bedroom', 
            'price_per_bathroom', 'total_rooms', 'desirability_score',
            'type_encoded', 'locality_encoded', 'state_encoded',
            'locality_avg_price', 'locality_median_price', 'locality_price_std'
        ]
        
        # Create feature matrix
        self.X = self.df[feature_columns].copy()
        
        # Handle any remaining NaN values
        self.X = self.X.fillna(self.X.median())
        
        # Target variable (log-transformed for better model performance)
        self.y = np.log1p(self.df['price'])
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Features used: {list(self.X.columns)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        
        return self.X, self.y
    
    def train_baseline_models(self):
        """Train baseline linear models"""
        print("\n" + "="*50)
        print("TRAINING BASELINE MODELS")
        print("="*50)
        
        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        for name, model in baseline_models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Convert back from log scale
            y_pred_train_orig = np.expm1(y_pred_train)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_train_orig = np.expm1(self.y_train)
            y_test_orig = np.expm1(self.y_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
            train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
            test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
            train_r2 = r2_score(y_train_orig, y_pred_train_orig)
            test_r2 = r2_score(y_test_orig, y_pred_test_orig)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                     cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'y_pred_test': y_pred_test_orig
            }
            
            print(f"  Train RMSE: ${train_rmse:,.0f}")
            print(f"  Test RMSE: ${test_rmse:,.0f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  CV RMSE: ${cv_rmse:,.0f}")
    
    def train_advanced_models(self):
        """Train advanced ensemble models"""
        print("\n" + "="*50)
        print("TRAINING ADVANCED MODELS")
        print("="*50)
        
        advanced_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
        }
        
        for name, model in advanced_models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Convert back from log scale
            y_pred_train_orig = np.expm1(y_pred_train)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_train_orig = np.expm1(self.y_train)
            y_test_orig = np.expm1(self.y_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
            train_mae = mean_absolute_error(y_pred_train_orig, y_train_orig)
            test_mae = mean_absolute_error(y_pred_test_orig, y_test_orig)
            train_r2 = r2_score(y_train_orig, y_pred_train_orig)
            test_r2 = r2_score(y_test_orig, y_pred_test_orig)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                     cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_rmse,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'y_pred_test': y_pred_test_orig
            }
            
            print(f"  Train RMSE: ${train_rmse:,.0f}")
            print(f"  Test RMSE: ${test_rmse:,.0f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  CV RMSE: ${cv_rmse:,.0f}")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Random Forest tuning
        print("\nTuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        # XGBoost tuning
        print("Tuning XGBoost...")
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            xgb_param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        xgb_grid.fit(self.X_train, self.y_train)
        
        # Store best models
        self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
        self.models['XGBoost (Tuned)'] = xgb_grid.best_estimator_
        
        print(f"Best RF params: {rf_grid.best_params_}")
        print(f"Best XGB params: {xgb_grid.best_params_}")
        
        # Evaluate tuned models
        for name in ['Random Forest (Tuned)', 'XGBoost (Tuned)']:
            model = self.models[name]
            
            # Predictions
            y_pred_test = model.predict(self.X_test)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_test_orig = np.expm1(self.y_test)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
            test_mae = mean_absolute_error(y_pred_test_orig, y_test_orig)
            test_r2 = r2_score(y_test_orig, y_pred_test_orig)
            
            self.results[name] = {
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'y_pred_test': y_pred_test_orig
            }
            
            print(f"\n{name} Results:")
            print(f"  Test RMSE: ${test_rmse:,.0f}")
            print(f"  Test R²: {test_r2:.4f}")
    
    def compare_models(self):
        """Compare all models and select the best one"""
        print("\n" + "="*60)
        print("MODEL COMPARISON & SELECTION")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Test RMSE': f"${results['test_rmse']:,.0f}",
                'Test R²': f"{results['test_r2']:.4f}",
                'Test MAE': f"${results['test_mae']:,.0f}" if 'test_mae' in results else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        best_results = self.results[best_model_name]
        print(f"   Test RMSE: ${best_results['test_rmse']:,.0f}")
        print(f"   Test R²: {best_results['test_r2']:.4f}")
        print(f"   Test MAE: ${best_results['test_mae']:,.0f}")
        
        # Save best model
        joblib.dump(best_model, '/home/aron/code/project/best_price_model.pkl')
        joblib.dump(self.scaler, '/home/aron/code/project/feature_scaler.pkl')
        
        return best_model, best_model_name
    
    def create_visualizations(self):
        """Create model performance visualizations"""
        print("\nCreating model performance visualizations...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Chart
        models = list(self.results.keys())
        rmse_values = [self.results[model]['test_rmse'] for model in models]
        
        axes[0, 0].bar(range(len(models)), rmse_values, color='skyblue')
        axes[0, 0].set_title('Model Comparison - Test RMSE')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('RMSE ($)')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # 2. R² Score Comparison
        r2_values = [self.results[model]['test_r2'] for model in models]
        
        axes[0, 1].bar(range(len(models)), r2_values, color='lightgreen')
        axes[0, 1].set_title('Model Comparison - R² Score')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Feature Importance (for tree-based models)
        tree_models = ['Random Forest', 'XGBoost', 'Random Forest (Tuned)', 'XGBoost (Tuned)']
        available_tree_models = [model for model in tree_models if model in self.models]
        
        if available_tree_models:
            best_tree_model = available_tree_models[0]
            model = self.models[best_tree_model]
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                feature_names = self.X.columns
                
                # Sort features by importance
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=True)
                
                axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
                axes[1, 0].set_title(f'Feature Importance - {best_tree_model}')
                axes[1, 0].set_xlabel('Importance')
        
        # 4. Prediction vs Actual (for best model)
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_r2'])
        predictions = self.results[best_model_name]['y_pred_test']
        actuals = np.expm1(self.y_test)
        
        axes[1, 1].scatter(actuals, predictions, alpha=0.6)
        axes[1, 1].plot([actuals.min(), actuals.max()], 
                        [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Price ($)')
        axes[1, 1].set_ylabel('Predicted Price ($)')
        axes[1, 1].set_title(f'Predictions vs Actual - {best_model_name}')
        
        plt.tight_layout()
        plt.savefig('/home/aron/code/project/model_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model_name
    
    def run_complete_pipeline(self):
        """Run the complete modeling pipeline"""
        try:
            # Load and prepare data
            self.load_data()
            self.prepare_features()
            
            # Train models
            self.train_baseline_models()
            self.train_advanced_models()
            self.hyperparameter_tuning()
            
            # Compare and select best model
            best_model, best_model_name = self.compare_models()
            
            # Create visualizations
            best_model_name = self.create_visualizations()
            
            print("\n🎉 PHASE 5 COMPLETED SUCCESSFULLY!")
            print(f"Best model: {best_model_name}")
            print("Model saved as 'best_price_model.pkl'")
            print("Visualization saved as 'model_performance.png'")
            
            return best_model, best_model_name
            
        except Exception as e:
            print(f"❌ Error in modeling pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main execution function"""
    print("🤖 Starting Machine Learning Model Development")
    print("="*60)
    
    # Initialize the model trainer
    model_trainer = PricePredictionModels('/home/aron/code/project/cleaned_real_estate_data.csv')
    
    # Run complete pipeline
    best_model, best_model_name = model_trainer.run_complete_pipeline()
    
    return model_trainer, best_model, best_model_name

if __name__ == "__main__":
    model_trainer, best_model, best_model_name = main()
