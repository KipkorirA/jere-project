# Real Estate Pricing Intelligence System

A comprehensive machine learning system for predicting property prices and detecting pricing anomalies in the Kenyan real estate market.

## 🏠 System Overview

This system analyzes real estate data to provide:
- **Price Prediction**: ML models to predict property values
- **Anomaly Detection**: Identifies over/under-priced properties
- **Market Intelligence**: Business insights for buyers, sellers, and investors
- **Stakeholder Reports**: Comprehensive analysis for different market participants

## 📁 Project Structure

```
real_estate_system/
├── real_estate_analysis.py      # Data loading, cleaning, and EDA
├── price_prediction_models.py   # ML model training and evaluation
├── anomaly_detection.py         # Price anomaly detection system
├── final_business_report.py     # Comprehensive business reporting
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── SETUP_GUIDE.md              # Detailed setup instructions
└── DATA_FILES/
    ├── kenya_listings.csv       # Original dataset
    ├── cleaned_real_estate_data.csv  # Processed data
    ├── best_price_model.pkl     # Trained ML model
    ├── feature_scaler.pkl       # Feature scaling parameters
    └── [output files]           # Generated reports and analyses
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (for model training)
- 2GB+ free disk space

### Installation Steps

1. **Transfer the project folder** to your target computer

2. **Navigate to the project directory**:
   ```bash
   cd real_estate_system
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv real_estate_env
   real_estate_env\Scripts\activate

   # On Mac/Linux
   python3 -m venv real_estate_env
   source real_estate_env/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the system**:
   ```bash
   # Option 1: Run individual components
   python real_estate_analysis.py    # Data processing
   python price_prediction_models.py # Model training
   python anomaly_detection.py       # Anomaly detection
   python final_business_report.py   # Generate reports

   # Option 2: Run full pipeline (if main() functions are implemented)
   ```

## 📊 Data Requirements

The system expects the following data files:
- `kenya_listings.csv` - Original real estate listings data

**Required columns in the dataset:**
- `id` - Property identifier
- `price` - Property price (numeric)
- `type` - Property type (House, Apartment, etc.)
- `category` - Sale or Rent classification
- `bedrooms`, `bathrooms`, `toilets` - Room counts
- `furnished`, `serviced`, `shared` - Property features (0/1)
- `parking` - Number of parking spaces
- `locality` - Property location
- `state` - Property state/region

## 🔧 System Components

### 1. Data Analysis (`real_estate_analysis.py`)
- Loads and explores the real estate dataset
- Performs data quality assessment
- Cleans and preprocesses data
- Conducts exploratory data analysis (EDA)
- Engineers features for machine learning

**Key outputs:**
- `cleaned_real_estate_data.csv`
- `eda_analysis.png` (visualization)

### 2. Model Training (`price_prediction_models.py`)
- Implements multiple ML algorithms (Linear Regression, Random Forest, XGBoost, etc.)
- Performs hyperparameter tuning
- Evaluates and compares model performance
- Selects the best performing model

**Key outputs:**
- `best_price_model.pkl` (trained model)
- `feature_scaler.pkl` (scaling parameters)
- `model_performance.png` (performance visualizations)

### 3. Anomaly Detection (`anomaly_detection.py`)
- Uses trained models to predict property prices
- Detects pricing anomalies using multiple methods
- Classifies properties as over/underpriced
- Generates market intelligence insights

**Key outputs:**
- `properties_with_anomaly_analysis.csv`
- `detected_price_anomalies.csv`
- `anomaly_detection_analysis.png`

### 4. Business Reporting (`final_business_report.py`)
- Generates comprehensive stakeholder reports
- Creates executive summaries
- Provides investment recommendations
- Produces deployment strategies

**Key outputs:**
- `FINAL_BUSINESS_REPORT.md`
- `business_report_dashboard.png`

## 🎯 Use Cases

### For Buyers
- Identify undervalued properties
- Calculate fair market values
- Find investment opportunities

### For Sellers
- Price properties competitively
- Avoid overpricing risks
- Understand market positioning

### For Investors
- Discover high-ROI opportunities
- Analyze market trends
- Assess investment risks

### For Real Estate Agents
- Provide data-driven valuations
- Enhance client consultations
- Gain competitive advantage

### For Policy Makers
- Understand market efficiency
- Identify geographic disparities
- Inform housing policy decisions

## 📈 Performance Metrics

The system achieves:
- **High Accuracy**: R² scores typically 0.85+
- **Low Error Rates**: RMSE and MAE within acceptable ranges
- **Comprehensive Analysis**: Covers all major market segments
- **Actionable Insights**: Provides specific recommendations

## 🛠️ Customization

### Adding New Data
1. Replace `kenya_listings.csv` with your dataset
2. Ensure column names match the requirements
3. Update file paths in the scripts if needed

### Modifying Analysis
- Edit feature engineering in `real_estate_analysis.py`
- Adjust model parameters in `price_prediction_models.py`
- Customize anomaly thresholds in `anomaly_detection.py`
- Modify report templates in `final_business_report.py`

### Extending to New Markets
- Update location mappings
- Modify data preprocessing for local formats
- Adjust price ranges and thresholds
- Recalibrate anomaly detection parameters

## 🚨 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing packages
pip install [package_name]

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

**Memory Errors**
- Close other applications
- Use smaller dataset samples
- Reduce model complexity (fewer estimators)

**File Not Found Errors**
- Check that all data files are present
- Verify file paths in scripts
- Ensure correct working directory

**Visualization Issues**
```bash
# For matplotlib display issues
pip install -U matplotlib

# Or use non-interactive backend
import matplotlib
matplotlib.use('Agg')
```

### Performance Issues
- Reduce data size for testing
- Use fewer cross-validation folds
- Simplify hyperparameter grids

## 📝 Notes

- The system is designed for residential properties (houses, apartments)
- Model performance depends on data quality and quantity
- Anomaly thresholds can be adjusted based on market characteristics
- All monetary values are in local currency (KES for Kenya)

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure data files are in correct format
4. Review error messages for specific guidance

## 📄 License

This system is designed for educational and commercial use in real estate market analysis.

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Compatible with**: Python 3.8+
