
# 🚗 Accidental Analysis using Random Forest

A machine learning pipeline to predict accident frequency/severity based on historical accident and environmental data using the **Random Forest** algorithm.

---

## 🔍 Project Highlights

- 📈 Train a **Random Forest** classifier/regressor for accident prediction
- 🗂️ Includes data cleaning, feature engineering, model tuning & evaluation
- ✅ Comparison with baseline models (e.g. Decision Tree, KNN, Logistic Regression)
- 🌦️ Factor analysis to identify key influences like weather, time, location

---

## 🧰 Technologies Used

| Role           | Tools & Libraries                      |
|----------------|----------------------------------------|
| Language       | Python 3.x                             |
| Data Handling  | Pandas, NumPy                          |
| Modeling       | scikit-learn (RandomForestClassifier / Regressor) |
| Evaluation     | Matplotlib / Seaborn, scikit-learn metrics |
| Environment    | Jupyter Notebook / Google Colab        |

---

## 📂 Folder Structure

```
accidental-analysis-using-random-forest/
├── data/
│   ├── accidents.csv
│   └── weather.csv
├── notebooks/
│   └── accidental_analysis.ipynb
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   └── model_training.py
├── models/
│   └── rf_accident_model.pkl
├── README.md
└── requirements.txt
```

*(Adjust paths based on your actual structure)*

---

## 🚀 Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/gauthamkn/accidental-analysis-using-random-forest.git
cd accidental-analysis-using-random-forest
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the analysis notebook**

Launch `notebooks/accidental_analysis.ipynb` in Jupyter or Google Colab.

4. **Or run scripts**

```bash
python src/data_preparation.py
python src/feature_engineering.py
python src/model_training.py
```

This will clean data, engineer features, train the Random Forest model, evaluate it, and save the trained model in `models/`.

---

## 📊 Sample Results

- **Model Accuracy / R² Score**: e.g. ~82% / 0.78 (placeholder — replace with your actual results)
- **Feature Importance**: Presented in the notebook, showcasing top predictors like weather, time-of-day, traffic conditions, etc.
- **Evaluation Metrics**: Displaying classification report (precision, recall, F1) or regression metrics.

---

## 🧠 Methodology

1. **Data Cleaning**: Removed missing values, encoded categorical variables, merged related datasets  
2. **Feature Engineering**: Derived temporal features (hour, weekday), merged weather data  
3. **Modeling**: Trained Random Forest with hyperparameter tuning  
4. **Evaluation**: Compared with baseline models, evaluated using cross-validation  
5. **Insights**: Generated feature importance plots and partial dependence graphs

---

## 🎯 Next Steps

- 📦 Containerize the pipeline with Docker  
- 📡 Deploy model as an API (Flask / FastAPI)  
- 📊 Integrate geospatial visualization (folium / plotly map)  
- 🔍 Extend model comparison with XGBoost or LightGBM

---

## 🤝 Contributing

Contributions are welcome! Please fork, make updates, and open PRs or create issues for discussions.

---

> ⚠️ *Disclaimer:* This project is for educational/research purposes only.
