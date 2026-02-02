import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
import shap



# 1. Load the dataset
data = pd.read_csv(
    '/Local/NewProductSyntheticData.csv',
    sep=';',
    encoding='latin1'
)

data.columns = [c.lower() for c in data.columns]

# 2. Feature Engineering
for col in ['category', 'brand', 'color']:
    data[col] = data[col].astype('category').cat.codes

weeks = [f'demand{i:02d}' for i in range(1, 19)]

early_weeks = weeks[:12]
data['early_mean'] = data[early_weeks].mean(axis=1)
data['early_slope'] = data[early_weeks].diff(axis=1).iloc[:, 1:].mean(axis=1)

category_avg = data.groupby('category')[early_weeks].mean().reset_index()
data = data.merge(category_avg, on='category', suffixes=('', '_cat_avg'))

target_weeks = weeks[12:]  

feature_cols = ['category','brand','color','price','early_mean','early_slope'] + [f'{w}_cat_avg' for w in early_weeks]
X = data[feature_cols]

X_long = pd.concat([X]*len(target_weeks), ignore_index=True)

y_long = data[target_weeks].values.flatten()

X_long['week_index'] = np.tile(np.arange(1, len(target_weeks)+1), len(data))

X_train, X_test, y_train, y_test = train_test_split(X_long, y_long, test_size=0.2, random_state=42)

numeric_cols = ['price','early_mean','early_slope'] + [f'{w}_cat_avg' for w in early_weeks] + ['week_index']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

models = {
    'LinearRegression': LinearRegression(),
    'RidgeRegression': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
}

# 5. Train & Evaluate Models (Train vs Test)
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        'Model': name,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    })

    print(f"""
{name}
Train -> MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}
Test  -> MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}
""")

results_df = pd.DataFrame(results)
results_df



# --- 6. Updated Feature Importance Comparison (Including Linear Models) ---
# We use coefficients for linear models and feature_importances_ for tree models
for name in models.keys():
    model = models[name]
    feature_names = X_train.columns
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based importance
        importance = model.feature_importances_
        title_suffix = "Feature Importances"
    elif hasattr(model, 'coef_'):
        # Linear model importance (using absolute values of coefficients)
        importance = np.abs(model.coef_)
        title_suffix = "Absolute Coefficients"
    else:
        continue

    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=feat_imp.head(20), x='importance', y='feature', palette='magma' if 'Linear' in name else 'viridis')
    plt.title(f'{name} Top 20 {title_suffix}')
    plt.show()


# --- 7. Updated SHAP Analysis (Comprehensive Comparison) ---
# We use a 1000-sample subset to ensure consistency and reasonable runtime
sample_idx = np.random.choice(X_train.index, size=1000, replace=False)
X_shap = X_train.loc[sample_idx]

# Explaining all models to contrast their decision-making logic
explain_models = ["LinearRegression", "RidgeRegression", "RandomForest", "XGBoost", "LightGBM"]

for name in explain_models:
    model = models[name]
    plt.figure()
    
    if "Regression" in name:
        # Linear Explainer specifically for Linear and Ridge baselines
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_shap)
    else:
        # Tree Explainer for RandomForest, XGBoost, and LightGBM
        # Note: RandomForest SHAP can be slower due to tree depth and ensemble size
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
    
    # Customizing the title to reflect each model's specific SHAP distribution
    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )
    plt.title(f"{name} SHAP Summary Plot")
    plt.tight_layout()
    plt.show()


