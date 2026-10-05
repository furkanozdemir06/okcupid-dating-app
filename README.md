# OkCupid Dating App: Age Regression & Generation Classification

Machine learning project on 60,000+ OkCupid profiles with two tasks:

- **Regression:** predict a user's age
- **Classification:** predict a user's generational cohort (Millennial, Gen X-er, Boomer)

## Workflow

1. Exploratory data analysis (including an automated `ydata-profiling` report)
2. Data cleaning: dropped free-text essays, filtered outliers (age 18-80, height 55-82 in), handled missing values
3. Feature engineering: grouped categories (diet, job, education, ethnicity, pets), binary encoding, one-hot encoding, min-max scaling
4. Model comparison: classical ML models vs. deep neural networks (Keras)

## Results

| Task | Best Model | Score |
| :--- | :--- | :---: |
| Regression (age) | Gradient Boosting Regressor | R² 0.84 |
| Regression (age) | Deep Neural Network | R² 0.81 |
| Classification (generation) | Random Forest | Accuracy 0.87 |
| Classification (generation) | Deep Neural Network | Accuracy 0.84 |

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn, ydata-profiling

## Files

- `OKCupid_DatingApp.ipynb`: full analysis and modeling notebook
- `okcupid_analysis_report.html`: automated EDA report
- `model_performance_comparison.png`: model comparison chart

## Data

The OkCupid Profiles dataset is available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles). It is not included in this repository because it contains real user profile data. To run the notebook, download `profiles.csv` and place it in the project root.
