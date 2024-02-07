import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
#from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Generate synthetic credit card transaction data
np.random.seed(42)
n_samples = 10000

# Create features
data = {
    'transaction_amount': np.random.exponential(scale=100, size=n_samples),
    'transaction_hour': np.random.randint(0, 24, n_samples),
    'days_since_last_transaction': np.random.exponential(scale=5, size=n_samples),
    'distance_from_home': np.random.exponential(scale=50, size=n_samples),
    'distance_from_last_transaction': np.random.exponential(scale=30, size=n_samples),
    'ratio_to_median_purchase': np.random.lognormal(mean=0, sigma=1, size=n_samples),
    'used_chip': np.random.choice([0, 1], n_samples, p=[0.15, 0.85]),
    'used_pin': np.random.choice([0, 1], n_samples, p=[0.20, 0.80]),
    'online_order': np.random.choice([0, 1], n_samples, p=[0.70, 0.30])
}

df = pd.DataFrame(data)

# Generate fraud labels (imbalanced: ~2% fraud)
fraud_prob = 0.02 + (
    0.1 * (df['transaction_amount'] > 500) +
    0.05 * (df['transaction_hour'] < 6) +
    0.08 * (df['distance_from_home'] > 100) +
    0.06 * (df['used_chip'] == 0) +
    0.04 * (df['online_order'] == 1)
)
fraud_prob = np.clip(fraud_prob, 0, 0.3)
df['is_fraud'] = np.random.binomial(1, fraud_prob)

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("=" * 60)
print(f"\nDataset Overview:")
print(f"Total Transactions: {len(df):,}")
print(f"Fraud Cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"Legitimate Cases: {(~df['is_fraud'].astype(bool)).sum():,} ({(1-df['is_fraud'].mean())*100:.2f}%)")

# Display sample data
print("\n" + "-" * 60)
print("Sample Transactions:")
print("-" * 60)
print(df.head(10))

# Statistical summary
print("\n" + "-" * 60)
print("Statistical Summary:")
print("-" * 60)
print(df.describe())

# Fraud vs Legitimate comparison
print("\n" + "-" * 60)
print("Fraud vs Legitimate Transaction Comparison:")
print("-" * 60)
fraud_comparison = df.groupby('is_fraud').agg({
    'transaction_amount': ['mean', 'median'],
    'distance_from_home': ['mean', 'median'],
    'transaction_hour': 'mean',
    'used_chip': 'mean',
    'online_order': 'mean'
})
print(fraud_comparison)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Fraud Detection - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Transaction amount distribution
axes[0, 0].hist(df[df['is_fraud']==0]['transaction_amount'], bins=50, alpha=0.7, label='Legitimate', color='blue')
axes[0, 0].hist(df[df['is_fraud']==1]['transaction_amount'], bins=50, alpha=0.7, label='Fraud', color='red')
axes[0, 0].set_xlabel('Transaction Amount ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Transaction Amount Distribution')
axes[0, 0].legend()

# Transaction hour distribution
axes[0, 1].hist(df[df['is_fraud']==0]['transaction_hour'], bins=24, alpha=0.7, label='Legitimate', color='blue')
axes[0, 1].hist(df[df['is_fraud']==1]['transaction_hour'], bins=24, alpha=0.7, label='Fraud', color='red')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Transaction Hour Distribution')
axes[0, 1].legend()

# Distance from home
axes[0, 2].hist(df[df['is_fraud']==0]['distance_from_home'], bins=50, alpha=0.7, label='Legitimate', color='blue')
axes[0, 2].hist(df[df['is_fraud']==1]['distance_from_home'], bins=50, alpha=0.7, label='Fraud', color='red')
axes[0, 2].set_xlabel('Distance from Home (km)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distance from Home Distribution')
axes[0, 2].legend()

# Fraud by chip usage
chip_fraud = df.groupby(['used_chip', 'is_fraud']).size().unstack()
chip_fraud.plot(kind='bar', ax=axes[1, 0], color=['blue', 'red'], alpha=0.7)
axes[1, 0].set_xlabel('Chip Used')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Fraud by Chip Usage')
axes[1, 0].set_xticklabels(['No', 'Yes'], rotation=0)
axes[1, 0].legend(['Legitimate', 'Fraud'])

# Fraud by online order
online_fraud = df.groupby(['online_order', 'is_fraud']).size().unstack()
online_fraud.plot(kind='bar', ax=axes[1, 1], color=['blue', 'red'], alpha=0.7)
axes[1, 1].set_xlabel('Online Order')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Fraud by Order Type')
axes[1, 1].set_xticklabels(['In-Store', 'Online'], rotation=0)
axes[1, 1].legend(['Legitimate', 'Fraud'])

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 2])
axes[1, 2].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# Prepare data for modeling
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)
print(f"\nTraining set (after SMOTE): {len(X_train_balanced):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Train Random Forest
print("\n" + "-" * 60)
print("Training Random Forest Classifier...")
print("-" * 60)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraud']))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Train Logistic Regression
print("\n" + "-" * 60)
print("Training Logistic Regression...")
print("-" * 60)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_balanced, y_train_balanced)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr, target_names=['Legitimate', 'Fraud']))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "-" * 60)
print("Feature Importance (Random Forest):")
print("-" * 60)
print(feature_importance)

# Visualization of results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix - Random Forest')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix - Logistic Regression')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# ROC Curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)

axes[1, 0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_pred_proba_rf):.3f})', linewidth=2)
axes[1, 0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_pred_proba_lr):.3f})', linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curves Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Feature Importance
feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=axes[1, 1], legend=False, color='steelblue')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_title('Feature Importance (Random Forest)')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
sample_indices = np.random.choice(range(len(X_test)), 5, replace=False)
sample_predictions = pd.DataFrame({
    'Amount': X_test.iloc[sample_indices]['transaction_amount'].values,
    'Hour': X_test.iloc[sample_indices]['transaction_hour'].values,
    'Distance_Home': X_test.iloc[sample_indices]['distance_from_home'].values,
    'Actual': y_test.iloc[sample_indices].values,
    'RF_Prediction': y_pred_rf[sample_indices],
    'RF_Probability': y_pred_proba_rf[sample_indices],
    'LR_Prediction': y_pred_lr[sample_indices],
    'LR_Probability': y_pred_proba_lr[sample_indices]
})
print(sample_predictions.to_string())

print("\n" + "=" * 60)
print("FRAUD DETECTION SYSTEM COMPLETE")
print("=" * 60)