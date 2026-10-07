"""
=============================================================================
MODULE: regression.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Automated training, evaluation, and comparison of Regression algorithms:
1. Decision Tree Regressor
2. Random Forest Regressor
3. K-Nearest Neighbors Regressor

Metrics: MAE, MSE, RMSE, R² Score
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def train_and_evaluate_regressors(X_train, X_test, y_train, y_test):
    """
    Trains and compares Decision Tree, Random Forest, and KNN on any regression task.
    """
    n_samples = len(X_train)
    k_val = min(7, max(1, n_samples // 5))
    
    models = {
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=42
        ),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100,
            max_depth=9,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        'KNN Regressor': KNeighborsRegressor(
            n_neighbors=k_val,
            weights='distance',
            metric='minkowski'
        )
    }
    
    metrics = {}
    trained_models = {}
    test_predictions = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics[name] = {
            'MAE': round(mae, 3),
            'MSE': round(mse, 3),
            'RMSE': round(rmse, 3),
            'R2_Score': round(r2, 3)
        }
        trained_models[name] = model
        test_predictions[name] = {
            'actual': y_test.tolist(),
            'predicted': y_pred.tolist()
        }
        
    # Select best model based on highest R2 Score (or lowest RMSE)
    best_model_name = max(metrics, key=lambda m: (metrics[m]['R2_Score'], -metrics[m]['RMSE']))
    best_score = metrics[best_model_name]['R2_Score']
    
    return metrics, trained_models, test_predictions, best_model_name, best_score
