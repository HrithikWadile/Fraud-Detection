# Fraud-Detection
🕵️‍♂️ Credit Card Fraud Detection System — Machine Learning Project
This project demonstrates a complete end-to-end Credit Card Fraud Detection System built using Python and Machine Learning.
It includes synthetic data generation, exploratory data analysis (EDA), feature engineering, model training, class imbalance handling, evaluation, and visualization.
Fraud detection is a highly imbalanced binary classification problem, and this project focuses on building interpretable and high-precision models to detect fraudulent transactions.

🚀 Project Highlights
✔️ 1. Synthetic Transaction Dataset
A dataset of 10,000 simulated transactions was generated to mimic real-world credit card behavior.
Features include:
Transaction amount
Hour of transaction
Distance from home & last transaction
Chip usage
PIN usage
Online vs in-store behavior
Ratio to median purchase
Days since last transaction
Fraud probability is generated using realistic patterns (high amounts, late-night usage, no chip, unusual distances).

🔍 2. Exploratory Data Analysis (EDA)
The project includes deep EDA with:
Distribution plots of transaction amount, hour, distance, etc.
Fraud vs legitimate comparison
Chip and online-order fraud patterns
Correlation heatmap
Feature-level insights
The analysis highlights how fraud differs from normal transactions.


🤖 3. Machine Learning Models Used
Two ML models are trained and evaluated:
🔹 Random Forest Classifier
Handles non-linear relationships
Interpretable via feature importance
Performs strongly on noisy data

🔹 Logistic Regression
Interpretable baseline model
Works well on scaled data
Useful for probability-based decisions

📈 4. Key Insights from the Model
High transaction amount & unusual distances are strong fraud indicators.
Late-night transactions increase fraud likelihood.
Fraud is more common when chip is not used.
Online transactions have a higher fraud probability.
These insights demonstrate real-world fraud patterns used in financial institutions.


ScreenShot:
<img width="1873" height="1078" alt="image" src="https://github.com/user-attachments/assets/e5d2e210-468e-4656-bda2-3dfbf2fc730f" />

