# Hospital Queue Prediction System

A machine learning project to predict wait times and queue lengths in hospital emergency rooms and departments.

## 📋 Project Overview

This project uses historical hospital data to forecast:
- **Queue Length**: Number of patients waiting
- **Wait Time**: Estimated time for patient to be seen
- **Peak Hours**: High-traffic periods
- **Resource Allocation**: Optimal staffing levels

## 🎯 Use Cases

- Emergency Room (ER) wait time prediction
- Out-patient department (OPD) queue management
- Resource planning and staff scheduling
- Patient communication (expected wait times)
- Hospital capacity planning

## 📁 Project Structure

```
queue-predict-hospital/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── hospital_queue_data.csv (sample)
│   └── processed/
│       └── cleaned_data.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── models/
│   └── trained_model.pkl
├── config/
│   └── config.yaml
└── tests/
    └── test_models.py
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sravanthipunagani559-ux/queue-predict-hospital.git
cd queue-predict-hospital
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- Place your hospital queue data in `data/raw/`
- Run preprocessing notebook or script

### 4. Train Models
```bash
python src/models.py
```

### 5. Make Predictions
```python
from src.models import predict_queue
wait_time = predict_queue(hour=14, day_of_week=3, department='ER')
```

## 📊 Data Requirements

Your dataset should include:

| Column | Description | Type |
|--------|-------------|------|
| timestamp | Date and time | DateTime |
| arrival_time | Patient arrival time | DateTime |
| service_start_time | When service began | DateTime |
| service_end_time | When service completed | DateTime |
| department | Hospital department (ER, OPD, etc.) | String |
| queue_length | Number of patients waiting | Integer |
| wait_time | Time patient waited (minutes) | Integer |
| num_doctors | Number of doctors on duty | Integer |
| num_nurses | Number of nurses on duty | Integer |
| patient_severity | Severity level (critical, urgent, routine) | String |

## 🤖 Models Implemented

### 1. **Time Series Models**
- ARIMA/SARIMA
- Prophet (Facebook's forecasting tool)
- LSTM (Deep Learning)

### 2. **Machine Learning Regression**
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)
- Linear Regression
- Support Vector Regression

### 3. **Queueing Theory**
- M/M/1 (Single server)
- M/M/c (Multiple servers)
- M/D/1 (Deterministic service)

## 📈 Key Features

- **Hourly Predictions**: Queue length for next hour
- **Daily Forecasts**: Peak times and average wait times
- **Department-wise Analysis**: Different departments have different patterns
- **Severity-based Queuing**: Triage levels affect wait times
- **Staff Optimization**: Recommendations for staffing levels

## 🔧 Technologies Used

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM
- **Time Series**: Statsmodels, Prophet
- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Flask/FastAPI (for deployment)
- **Deployment**: Docker, Streamlit

## 📚 Notebooks

1. **01_eda.ipynb** - Exploratory Data Analysis
2. **02_preprocessing.ipynb** - Data Cleaning & Feature Engineering
3. **03_model_training.ipynb** - Model Selection & Evaluation

## 🔍 Model Performance

Expected Metrics:
- **MAE (Mean Absolute Error)**: < 5 minutes
- **RMSE (Root Mean Squared Error)**: < 8 minutes
- **R² Score**: > 0.85

## 🚢 Deployment

### Docker
```bash
docker build -t queue-predict-hospital .
docker run -p 5000:5000 queue-predict-hospital
```

### Streamlit
```bash
streamlit run app.py
```

## 📞 API Endpoints

```bash
# Predict queue length
POST /predict
{
  "hour": 14,
  "day_of_week": 3,
  "department": "ER",
  "current_queue": 5
}

# Get historical trends
GET /trends?department=ER&days=7

# Get peak hours
GET /peak-hours?department=ER
```

## 📖 Documentation

- [Model Comparison Report](docs/model_comparison.md)
- [Data Dictionary](docs/data_dictionary.md)
- [API Guide](docs/api_guide.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sravanthi Punagani** (@sravanthipunagani559-ux)

## 📞 Support

For questions or issues, please open a GitHub Issue or contact the maintainer.

## 🎓 References

- [Time Series Forecasting](https://otexts.com/fpp2/)
- [Queueing Theory](https://en.wikipedia.org/wiki/Queueing_theory)
- [Hospital Operations Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4172088/)
- [Machine Learning for Healthcare](https://arxiv.org/abs/1902.01155)

---

**Last Updated**: 2026-08-31
