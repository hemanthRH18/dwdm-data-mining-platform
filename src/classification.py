"""
=============================================================================
MODULE: classification.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Automated training, evaluation, and comparison of Classification algorithms:
1. Decision Tree (CART)
2. Random Forest (Bagging Ensemble)
3. K-Nearest Neighbors (KNN)

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, class_names=None):
    """
    Trains and compares Decision Tree, Random Forest, and KNN on any classification task.
    """
    n_samples = len(X_train)
    k_val = min(7, max(1, n_samples // 5))
    if k_val % 2 == 0:
        k_val = max(1, k_val - 1)
        
    models = {
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=9,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=k_val,
            weights='distance',
            metric='minkowski'
        )
    }
    
    metrics = {}
    trained_models = {}
    confusion_matrices = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        metrics[name] = {
            'Accuracy': round(acc * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall': round(rec * 100, 2),
            'F1_Score': round(f1 * 100, 2)
        }
        trained_models[name] = model
        confusion_matrices[name] = cm
        
    # Select best model based on Accuracy (tie-break on F1)
    best_model_name = max(metrics, key=lambda m: (metrics[m]['Accuracy'], metrics[m]['F1_Score']))
    best_accuracy = metrics[best_model_name]['Accuracy']
    
    return metrics, trained_models, confusion_matrices, best_model_name, best_accuracy
