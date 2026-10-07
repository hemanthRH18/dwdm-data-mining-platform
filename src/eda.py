"""
=============================================================================
MODULE: eda.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE
=============================================================================

Dynamic Exploratory Data Analysis (EDA) visualizations that adapt
automatically to ANY uploaded CSV dataset and ANY selected target variable:
1. Target Variable Distribution (Classification & Regression)
2. Missing Value Analysis Chart
3. Pearson Correlation Heatmap for numeric attributes
4. Dynamic Feature-vs-Target Relationship Plots
5. Algorithm Comparison Bar Charts (Classification & Regression)
6. Diagnostic Plots: Confusion Matrices & Actual vs Predicted Residual Plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_target_distribution(df, target_col, problem_type='Classification'):
    """
    Visualizes the target distribution.
    - Classification: Countplot & Donut Chart
    - Regression: Histogram with KDE curve & Boxplot
    """
    if target_col not in df.columns:
        return None
        
    series = df[target_col].dropna()
    
    if problem_type == 'Classification':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        counts = series.value_counts()
        
        # Limit to top 15 classes if very high cardinality
        if len(counts) > 15:
            counts = counts.iloc[:15]
            
        palette = sns.color_palette("viridis", len(counts))
        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            hue=counts.index.astype(str),
            palette=palette,
            legend=False,
            edgecolor='black',
            linewidth=0.8,
            ax=ax1
        )
        ax1.set_title(f"Class Frequency: '{target_col}'", fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Target Class", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.tick_params(axis='x', rotation=45 if len(counts) > 5 else 0)
        
        # Pie / Donut
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct='%1.1f%%',
            startangle=140,
            colors=palette,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(9)
        ax2.set_title("Class Proportion (%)", fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout()
        return fig
        
    else:  # Regression
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        sns.histplot(numeric_series, kde=True, color='#1d3557', edgecolor='black', ax=ax1)
        ax1.set_title(f"Histogram & KDE: '{target_col}'", fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel(target_col, fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        
        sns.boxplot(x=numeric_series, color='#457b9d', ax=ax2)
        ax2.set_title(f"Boxplot: '{target_col}'", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel(target_col, fontsize=10)
        
        plt.tight_layout()
        return fig

def plot_missing_values(df):
    """
    Bar plot showing missing value counts and percentages across columns.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        return None  # No missing values
        
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    pct = (missing / len(df)) * 100
    
    sns.barplot(x=missing.values, y=missing.index, color='#e63946', edgecolor='black', ax=ax)
    ax.set_title("Missing Values Count by Column", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Number of Missing Records", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    
    for i, (cnt, p) in enumerate(zip(missing.values, pct.values)):
        ax.text(cnt + 1, i, f"{cnt} ({p:.1f}%)", va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, numeric_cols):
    """
    Pearson Correlation Heatmap across numeric features.
    """
    if len(numeric_cols) < 2:
        return None
        
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if len(valid_cols) < 2:
        return None
        
    corr = df[valid_cols].corr()
    fig, ax = plt.subplots(figsize=(max(7, len(valid_cols) * 0.8), max(6, len(valid_cols) * 0.7)))
    
    sns.heatmap(
        corr,
        annot=len(valid_cols) <= 15,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=12, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig

def plot_feature_vs_target(df, feature_col, target_col, problem_type='Classification'):
    """
    Bivariate analysis plot showing the relationship between any selected feature and target.
    """
    if feature_col not in df.columns or target_col not in df.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_is_numeric = pd.api.types.is_numeric_dtype(df[feature_col])
    
    if problem_type == 'Classification':
        if feat_is_numeric:
            sns.boxplot(
                x=target_col,
                y=feature_col,
                data=df,
                hue=target_col,
                palette='Set2',
                legend=False,
                ax=ax
            )
            ax.set_title(f"'{feature_col}' Distribution across Target Classes", fontsize=12, fontweight='bold', pad=10)
        else:
            # Categorical feature vs Classification target
            counts = df.groupby([feature_col, target_col]).size().unstack(fill_value=0)
            counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title(f"'{feature_col}' vs. '{target_col}' (Stacked)", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            ax.legend(title=target_col)
    else:  # Regression
        if feat_is_numeric:
            sns.regplot(
                x=feature_col,
                y=target_col,
                data=df,
                scatter_kws={'alpha': 0.5, 'color': '#1d3557'},
                line_kws={'color': '#e63946'},
                ax=ax
            )
            ax.set_title(f"Scatter & Trend: '{feature_col}' vs. '{target_col}'", fontsize=12, fontweight='bold', pad=10)
        else:
            sns.boxplot(
                x=feature_col,
                y=target_col,
                data=df,
                hue=feature_col,
                palette='Set3',
                legend=False,
                ax=ax
            )
            ax.set_title(f"Target '{target_col}' by Category '{feature_col}'", fontsize=12, fontweight='bold', pad=10)
            plt.xticks(rotation=45, ha='right')
            
    plt.tight_layout()
    return fig

def plot_classification_comparison(metrics_dict):
    """
    Grouped bar chart for Classification metrics (Accuracy, Precision, Recall, F1).
    """
    df_metrics = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Algorithm'})
    melted = pd.melt(
        df_metrics,
        id_vars=['Algorithm'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
        var_name='Metric',
        value_name='Score (%)'
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {'Accuracy': '#1d3557', 'Precision': '#457b9d', 'Recall': '#2a9d8f', 'F1_Score': '#e76f51'}
    
    sns.barplot(data=melted, x='Algorithm', y='Score (%)', hue='Metric', palette=palette, edgecolor='black', ax=ax)
    ax.set_ylim(max(0, melted['Score (%)'].min() - 15), 105)
    ax.set_title("Algorithm Performance Comparison (Classification)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_regression_comparison(metrics_dict):
    """
    Comparison chart for Regression metrics (R2 Score, RMSE, MAE).
    """
    df_metrics = pd.DataFrame(metrics_dict).T.reset_index().rename(columns={'index': 'Algorithm'})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # R2 Score Plot
    sns.barplot(data=df_metrics, x='Algorithm', y='R2_Score', color='#2a9d8f', edgecolor='black', ax=ax1)
    ax1.set_title("R² Score Comparison (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("R² Score")
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points', fontweight='bold')
        
    # Error Metrics Plot (RMSE & MAE)
    melted_err = pd.melt(df_metrics, id_vars=['Algorithm'], value_vars=['RMSE', 'MAE'], var_name='Error Metric', value_name='Error Value')
    sns.barplot(data=melted_err, x='Algorithm', y='Error Value', hue='Error Metric', palette={'RMSE': '#e63946', 'MAE': '#f4a261'}, edgecolor='black', ax=ax2)
    ax2.set_title("Error Comparison: RMSE & MAE (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm_array, class_names, title="Confusion Matrix"):
    """
    Heatmap for classification confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = class_names if class_names and len(class_names) == len(cm_array) else True
    
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor='black',
        ax=ax
    )
    ax.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight='bold')
    ax.set_ylabel("Actual Ground Truth", fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_regression_actual_vs_predicted(actual, predicted, title="Actual vs. Predicted"):
    """
    Scatter plot comparing actual ground truth vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.6, color='#1d3557', edgecolors='white', linewidth=0.5)
    
    # 45-degree reference line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], color='#e63946', linestyle='--', linewidth=2, label="Ideal Fit (y = x)")
    
    ax.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Actual Value", fontsize=10, fontweight='bold')
    ax.set_ylabel("Predicted Value", fontsize=10, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig
