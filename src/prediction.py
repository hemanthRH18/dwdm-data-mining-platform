"""
=============================================================================
MODULE: prediction.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Handles live real-time model inference for any arbitrary user inputs,
supporting both Classification and Regression tasks.
"""

import numpy as np
import pandas as pd
from src.preprocessing import transform_user_input

def predict_sample(input_dict, model, preprocessor_bundle, model_name="Best Model"):
    """
    Performs inference on a single user input record.
    """
    problem_type = preprocessor_bundle['problem_type']
    target_col = preprocessor_bundle['target_col']
    
    # 1. Transform raw user input using fitted preprocessing artifacts (no data leakage)
    X_scaled = transform_user_input(input_dict, preprocessor_bundle)
    
    # 2. Perform Model Inference
    raw_pred = model.predict(X_scaled)[0]
    
    if problem_type == 'Classification':
        label_encoder = preprocessor_bundle['label_encoder']
        class_names = preprocessor_bundle['class_names']
        
        # Decode predicted class
        if label_encoder is not None:
            predicted_label = label_encoder.inverse_transform([int(raw_pred)])[0]
        else:
            predicted_label = str(raw_pred)
            
        # Calculate probabilities if supported
        probabilities = {}
        confidence = 100.0
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            for idx, p in enumerate(probs):
                cls_name = class_names[idx] if idx < len(class_names) else str(idx)
                probabilities[cls_name] = round(float(p) * 100, 1)
            confidence = probabilities.get(predicted_label, round(float(np.max(probs)) * 100, 1))
        else:
            probabilities = {predicted_label: 100.0}
            
        return {
            'problem_type': 'Classification',
            'target_col': target_col,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_name': model_name
        }
        
    else:  # Regression
        predicted_val = round(float(raw_pred), 3)
        return {
            'problem_type': 'Regression',
            'target_col': target_col,
            'prediction': predicted_val,
            'model_name': model_name
        }
