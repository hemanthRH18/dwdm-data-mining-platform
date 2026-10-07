"""
=============================================================================
MODULE: data_loader.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Handles dynamic loading of ANY CSV file (via uploaded file object or file path),
metadata extraction, automatic column type classification, and task inference.
"""

import io
import pandas as pd
import numpy as np

def load_csv_data(file_source):
    """
    Loads a CSV dataset dynamically from either:
    - Streamlit UploadedFile object
    - File path string
    - BytesIO / StringIO buffer
    Automatically detects delimiters (comma, semicolon, tab).
    """
    if file_source is None:
        raise ValueError("No file provided.")
        
    try:
        # If it's a Streamlit UploadedFile or file-like object
        if hasattr(file_source, 'read'):
            # Read first chunk to inspect
            raw_bytes = file_source.read()
            if isinstance(raw_bytes, bytes):
                try:
                    text_content = raw_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = raw_bytes.decode('latin-1')
            else:
                text_content = str(raw_bytes)
                
            # Try parsing with comma first, then semicolon, then tab
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(io.StringIO(text_content), sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            # Fallback default read
            return pd.read_csv(io.StringIO(text_content))
            
        elif isinstance(file_source, str):
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_source, sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            return pd.read_csv(file_source)
            
        else:
            raise TypeError(f"Unsupported file source type: {type(file_source)}")
            
    except Exception as e:
        raise ValueError(f"Unable to parse CSV file: {str(e)}")

def get_dataset_overview(df):
    """
    Extracts comprehensive metadata for any uploaded dataset.
    """
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    
    # Identify numeric and categorical columns
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if col not in numeric_cols]
    
    missing_dict = df.isnull().sum().to_dict()
    total_missing = sum(missing_dict.values())
    duplicate_count = int(df.duplicated().sum())
    
    dtypes_dict = {col: str(df[col].dtype) for col in columns}
    
    return {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'columns': columns,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'missing_values': missing_dict,
        'total_missing': total_missing,
        'duplicate_count': duplicate_count,
        'dtypes': dtypes_dict
    }

def infer_problem_type(df, target_col):
    """
    Automatically infers whether a chosen target column represents:
    - 'Classification' (discrete classes, categories, or small number of integers)
    - 'Regression' (continuous numeric distribution)
    
    Returns: (problem_type, reason_str)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    series = df[target_col].dropna()
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    
    if not is_numeric:
        return 'Classification', f"Categorical/Text target with {n_unique} distinct classes."
        
    # If numeric, check cardinality
    if n_unique <= 10:
        return 'Classification', f"Numeric target with only {n_unique} distinct discrete values (Likely class labels)."
    else:
        return 'Regression', f"Continuous numeric target with {n_unique} unique values."
