"""
=============================================================================
MODULE: preprocessing.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Generic Preprocessing Pipeline for ANY uploaded CSV dataset:
1. Deduplication (Removes redundant rows).
2. Missing Value Imputation (Median for numeric, mode for categorical).
3. Constant Feature Pruning (Removes zero-variance columns).
4. Categorical Encoding (OneHotEncoder for features, LabelEncoder for classification targets).
5. Partitioning (Train/Test Split with stratification for classification).
6. Feature Standardization (StandardScaler fitted ONLY on training partition to eliminate Data Leakage).
7. Inference Vectorizer (Transforms dynamic user inputs for real-time predictions).
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_generic_dataset(
    df,
    target_col,
    feature_cols=None,
    problem_type='Classification',
    test_size=0.20,
    random_state=42
):
    """
    Applies an automated, leakage-free preprocessing pipeline to any DataFrame.
    """
    if df is None or len(df) == 0:
        raise ValueError("Dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in dataset.")
        
    # 1. Deduplication
    working_df = df.copy()
    initial_len = len(working_df)
    working_df = working_df.drop_duplicates().reset_index(drop=True)
    dedup_removed = initial_len - len(working_df)
    
    # Drop rows where target is NaN
    working_df = working_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(working_df) < 10:
        raise ValueError("Dataset contains too few valid records (< 10 rows) after removing target nulls.")
        
    # 2. Separate Target and Features
    y_raw = working_df[target_col]
    
    if feature_cols is None or len(feature_cols) == 0:
        X_raw = working_df.drop(columns=[target_col])
    else:
        # Use user-specified feature subset (excluding target)
        valid_features = [c for c in feature_cols if c in working_df.columns and c != target_col]
        if len(valid_features) == 0:
            raise ValueError("No valid predictor feature columns selected.")
        X_raw = working_df[valid_features]
        
    # 3. Prune constant / all-null columns
    valid_cols = []
    for col in X_raw.columns:
        if X_raw[col].nunique() > 1 and not X_raw[col].isnull().all():
            valid_cols.append(col)
            
    if len(valid_cols) == 0:
        raise ValueError("All selected features have zero variance or are completely empty.")
        
    X_raw = X_raw[valid_cols].copy()
    
    # Identify numeric and categorical feature columns
    num_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    
    # 4. Imputation
    num_imputer = None
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])
        
    cat_imputer = None
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_raw[cat_cols] = cat_imputer.fit_transform(X_raw[cat_cols].astype(str))
        
    # 5. Target Encoding / Formatting
    label_encoder = None
    class_names = None
    if problem_type == 'Classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_names = [str(cls) for cls in label_encoder.classes_]
        if len(class_names) < 2:
            raise ValueError(f"Classification requires at least 2 distinct target classes, but only found {len(class_names)}.")
    else:  # Regression
        try:
            y = pd.to_numeric(y_raw, errors='coerce').to_numpy()
            if np.isnan(y).any():
                valid_idx = ~np.isnan(y)
                X_raw = X_raw.iloc[valid_idx].reset_index(drop=True)
                y = y[valid_idx]
        except Exception as e:
            raise ValueError(f"Target column could not be parsed as numeric for regression: {str(e)}")
            
    # 6. Train-Test Split (BEFORE feature encoding & scaling to prevent leakage)
    stratify_split = None
    if problem_type == 'Classification':
        # Check if each class has at least 2 samples for stratification
        unique_classes, counts = np.unique(y, return_counts=True)
        if (counts >= 2).all() and len(unique_classes) > 1:
            stratify_split = y
            
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_split
    )
    
    # 7. Categorical Encoding (OneHotEncoder) & Scaling
    ohe = None
    encoded_feature_names = []
    
    X_train_num = X_train[num_cols].to_numpy() if len(num_cols) > 0 else np.empty((len(X_train), 0))
    X_test_num = X_test[num_cols].to_numpy() if len(num_cols) > 0 else np.empty((len(X_test), 0))
    
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = ohe.fit_transform(X_train[cat_cols])
        X_test_cat = ohe.transform(X_test[cat_cols])
        
        cat_encoded_names = list(ohe.get_feature_names_out(cat_cols))
        encoded_feature_names = num_cols + cat_encoded_names
        
        X_train_combined = np.hstack([X_train_num, X_train_cat])
        X_test_combined = np.hstack([X_test_num, X_test_cat])
    else:
        encoded_feature_names = num_cols
        X_train_combined = X_train_num
        X_test_combined = X_test_num
        
    # 8. Feature Scaling (Fitted STRICTLY on X_train_combined to prevent Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
    # Package Preprocessing Bundle
    preprocessor_bundle = {
        'target_col': target_col,
        'problem_type': problem_type,
        'original_features': list(X_raw.columns),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'ohe': ohe,
        'scaler': scaler,
        'encoded_feature_names': encoded_feature_names,
        'label_encoder': label_encoder,
        'class_names': class_names,
        'dedup_removed': dedup_removed
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor_bundle

def transform_user_input(input_dict, preprocessor_bundle):
    """
    Transforms arbitrary user inputs for dynamic real-time model inference.
    """
    num_cols = preprocessor_bundle['num_cols']
    cat_cols = preprocessor_bundle['cat_cols']
    num_imputer = preprocessor_bundle['num_imputer']
    cat_imputer = preprocessor_bundle['cat_imputer']
    ohe = preprocessor_bundle['ohe']
    scaler = preprocessor_bundle['scaler']
    
    # Build single-row DataFrame matching original training feature names
    row_df = pd.DataFrame([input_dict])
    
    # Process numeric columns
    if len(num_cols) > 0:
        # Fill missing keys if any
        for col in num_cols:
            if col not in row_df.columns:
                row_df[col] = np.nan
        row_df[num_cols] = row_df[num_cols].apply(pd.to_numeric, errors='coerce')
        row_num = num_imputer.transform(row_df[num_cols])
    else:
        row_num = np.empty((1, 0))
        
    # Process categorical columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            if col not in row_df.columns:
                row_df[col] = "missing"
        row_df[cat_cols] = cat_imputer.transform(row_df[cat_cols].astype(str))
        row_cat = ohe.transform(row_df[cat_cols])
        row_combined = np.hstack([row_num, row_cat])
    else:
        row_combined = row_num
        
    # Scale input
    row_scaled = scaler.transform(row_combined)
    return row_scaled
