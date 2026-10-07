"""
End-to-End Automated Test Script for Generic DWDM Platform.
Tests:
1. Classification Dataset (data/sample_classification.csv -> Target: Species)
2. Regression Dataset (data/sample_regression.csv -> Target: Disease_Progression)
3. Custom Edge-Case Simulation (Missing values, duplicates, constant columns)
4. Full workflow: CSV loading -> Preprocessing -> EDA plots -> Model Training -> Comparison -> Live Prediction
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_csv_data, get_dataset_overview, infer_problem_type
from src.preprocessing import preprocess_generic_dataset, transform_user_input
from src.classification import train_and_evaluate_classifiers
from src.regression import train_and_evaluate_regressors
from src.eda import (
    plot_target_distribution,
    plot_missing_values,
    plot_correlation_matrix,
    plot_feature_vs_target,
    plot_classification_comparison,
    plot_regression_comparison,
    plot_confusion_matrix,
    plot_regression_actual_vs_predicted
)
from src.prediction import predict_sample

def test_classification_pipeline():
    print("\n" + "=" * 65)
    print("TEST 1: CLASSIFICATION PIPELINE (data/sample_classification.csv)")
    print("=" * 65)
    
    csv_path = "data/sample_classification.csv"
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    df = load_csv_data(csv_path)
    print(f"Loaded dataset: {len(df)} rows, {df.shape[1]} columns.")
    
    overview = get_dataset_overview(df)
    target_col = "Species"
    prob_type, reason = infer_problem_type(df, target_col)
    print(f"Target: '{target_col}', Inferred Task: {prob_type} ({reason})")
    assert prob_type == "Classification"
    
    # Preprocessing
    feature_cols = [c for c in df.columns if c != target_col]
    X_train, X_test, y_train, y_test, prep = preprocess_generic_dataset(
        df=df, target_col=target_col, feature_cols=feature_cols, problem_type=prob_type, test_size=0.2, random_state=42
    )
    print(f"Preprocessed X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Encoded feature count: {len(prep['encoded_feature_names'])}")
    
    # Model Training
    metrics, models, cms, best_model, best_acc = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test, class_names=prep['class_names']
    )
    print(f"\nModel Comparison Results:")
    for m_name, res in metrics.items():
        print(f"  [{m_name}] Accuracy: {res['Accuracy']}%, Precision: {res['Precision']}%, Recall: {res['Recall']}%, F1: {res['F1_Score']}%")
    print(f"🏆 Best Classification Model: {best_model} ({best_acc}%)")
    
    # EDA Plots Check
    f1 = plot_target_distribution(df, target_col, prob_type)
    f2 = plot_correlation_matrix(df, overview['numeric_cols'])
    f3 = plot_classification_comparison(metrics)
    f4 = plot_confusion_matrix(np.array(cms[best_model]), prep['class_names'])
    assert f1 is not None and f3 is not None
    print("EDA Visualizations verified cleanly.")
    
    # Prediction Check
    sample_input = {col: df[col].iloc[0] for col in prep['original_features']}
    pred_res = predict_sample(sample_input, models[best_model], prep, model_name=best_model)
    print(f"Live Prediction Result: '{pred_res['prediction']}', Confidence: {pred_res['confidence']}%")
    assert 'prediction' in pred_res
    print("Classification pipeline: SUCCESS!")

def test_regression_pipeline():
    print("\n" + "=" * 65)
    print("TEST 2: REGRESSION PIPELINE (data/sample_regression.csv)")
    print("=" * 65)
    
    csv_path = "data/sample_regression.csv"
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    df = load_csv_data(csv_path)
    print(f"Loaded dataset: {len(df)} rows, {df.shape[1]} columns.")
    
    overview = get_dataset_overview(df)
    target_col = "Disease_Progression"
    prob_type, reason = infer_problem_type(df, target_col)
    print(f"Target: '{target_col}', Inferred Task: {prob_type} ({reason})")
    assert prob_type == "Regression"
    
    # Preprocessing
    feature_cols = [c for c in df.columns if c != target_col]
    X_train, X_test, y_train, y_test, prep = preprocess_generic_dataset(
        df=df, target_col=target_col, feature_cols=feature_cols, problem_type=prob_type, test_size=0.2, random_state=42
    )
    print(f"Preprocessed X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Encoded feature count: {len(prep['encoded_feature_names'])}")
    
    # Model Training
    metrics, models, test_preds, best_model, best_r2 = train_and_evaluate_regressors(
        X_train, X_test, y_train, y_test
    )
    print(f"\nModel Comparison Results:")
    for m_name, res in metrics.items():
        print(f"  [{m_name}] R2: {res['R2_Score']}, RMSE: {res['RMSE']}, MSE: {res['MSE']}, MAE: {res['MAE']}")
    print(f"🏆 Best Regression Model: {best_model} (R2: {best_r2})")
    
    # EDA Plots Check
    f1 = plot_target_distribution(df, target_col, prob_type)
    f2 = plot_correlation_matrix(df, overview['numeric_cols'])
    f3 = plot_regression_comparison(metrics)
    f4 = plot_regression_actual_vs_predicted(
        test_preds[best_model]['actual'], test_preds[best_model]['predicted']
    )
    assert f1 is not None and f3 is not None
    print("EDA Visualizations verified cleanly.")
    
    # Prediction Check
    sample_input = {col: df[col].iloc[0] for col in prep['original_features']}
    pred_res = predict_sample(sample_input, models[best_model], prep, model_name=best_model)
    print(f"Live Prediction Result: '{pred_res['prediction']}' (Target: {target_col})")
    assert 'prediction' in pred_res
    print("Regression pipeline: SUCCESS!")

def test_custom_dataset_edge_cases():
    print("\n" + "=" * 65)
    print("TEST 3: CUSTOM DATASET WITH MISSING VALUES, DUPLICATES & MIXED TYPES")
    print("=" * 65)
    
    # Generate arbitrary realistic custom tabular data
    custom_df = pd.DataFrame({
        'CustomerID': [f"ID_{i}" for i in range(120)],
        'Age': [20 + (i % 40) for i in range(120)],
        'Income': [30000 + (i * 500) for i in range(120)],
        'Education': ['Bachelors' if i % 2 == 0 else 'Masters' for i in range(120)],
        'Constant_Col': ['Static_Value'] * 120,  # Zero-variance column to test pruning
        'Churn': ['Yes' if i % 3 == 0 else 'No' for i in range(120)]
    })
    
    # Inject missing values
    custom_df.loc[5:8, 'Age'] = np.nan
    custom_df.loc[10:12, 'Income'] = np.nan
    custom_df.loc[15:16, 'Education'] = np.nan
    
    # Inject duplicates
    custom_df = pd.concat([custom_df, custom_df.iloc[:5]], ignore_index=True)
    
    overview = get_dataset_overview(custom_df)
    print(f"Custom dataset created: {len(custom_df)} rows, {overview['duplicate_count']} duplicates, {overview['total_missing']} missing.")
    
    target_col = 'Churn'
    prob_type, reason = infer_problem_type(custom_df, target_col)
    print(f"Target: '{target_col}', Inferred: {prob_type}")
    
    # Feature selection excluding CustomerID
    feature_cols = ['Age', 'Income', 'Education', 'Constant_Col']
    X_train, X_test, y_train, y_test, prep = preprocess_generic_dataset(
        df=custom_df, target_col=target_col, feature_cols=feature_cols, problem_type=prob_type, test_size=0.25
    )
    print(f"Successfully pruned Constant_Col. Features remaining: {prep['original_features']}")
    assert 'Constant_Col' not in prep['original_features'], "Constant column was not pruned!"
    
    metrics, models, cms, best_model, best_acc = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test, class_names=prep['class_names']
    )
    print(f"Trained custom models. Best: {best_model} ({best_acc}%)")
    
    sample_input = {'Age': 28, 'Income': 45000, 'Education': 'Bachelors'}
    pred = predict_sample(sample_input, models[best_model], prep, model_name=best_model)
    print(f"Custom Prediction: {pred['prediction']}, Confidence: {pred['confidence']}%")
    print("Edge-case custom dataset test: SUCCESS!")

if __name__ == '__main__':
    test_classification_pipeline()
    test_regression_pipeline()
    test_custom_dataset_edge_cases()
    print("\n" + "=" * 65)
    print("ALL GENERIC DATASET TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 65)
