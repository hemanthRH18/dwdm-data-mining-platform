"""
=============================================================================
APPLICATION: app.py
PROJECT: Generic Dataset Analysis and Machine Learning Platform
COURSE: Data Warehousing and Data Mining (DWDM) - BTech CSE Mini Project
=============================================================================
A 100% generic, dynamic Data Mining & Machine Learning platform that accepts
ANY uploaded CSV dataset, automatically audits the schema, auto-detects
Classification vs. Regression, handles leakage-free preprocessing,
generates dynamic EDA visualizations, benchmarks 3 ML algorithms,
and provides live dynamic inference.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_csv_data,
    get_dataset_overview,
    infer_problem_type
)
from src.preprocessing import (
    preprocess_generic_dataset,
    transform_user_input
)
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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Generic ML Platform | DWDM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .main-banner h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .main-banner p {
        font-size: 1.05rem;
        color: #e0e6ed;
        margin-bottom: 0;
    }
    
    .best-model-banner {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.25);
    }
    
    .pred-card {
        background: linear-gradient(135deg, #1d3557, #457b9d);
        color: white;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(29, 53, 87, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = "No dataset loaded"
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'problem_type' not in st.session_state:
    st.session_state['problem_type'] = None
if 'trained_artifacts' not in st.session_state:
    st.session_state['trained_artifacts'] = None

SAMPLE_CLS_PATH = os.path.join(PROJECT_ROOT, "data", "sample_classification.csv")
SAMPLE_REG_PATH = os.path.join(PROJECT_ROOT, "data", "sample_regression.csv")

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
st.sidebar.title("DWDM ML Platform")
st.sidebar.caption("Generic Data Mining & Predictive Modeling")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📁 1. Dataset Upload & Overview",
        "⚙️ 2. Setup & Model Training",
        "📊 3. Exploratory Data Analysis",
        "⚖️ 4. Algorithm Benchmarking",
        "🔮 5. Live Prediction Portal",
        "📘 6. About Platform"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Dataset Status")
if st.session_state['df'] is not None:
    st.sidebar.success(f"**Loaded:** {st.session_state['dataset_name']}")
    st.sidebar.write(f"• **Rows:** {len(st.session_state['df'])}")
    st.sidebar.write(f"• **Columns:** {st.session_state['df'].shape[1]}")
    if st.session_state['target_col']:
        st.sidebar.write(f"• **Target:** `{st.session_state['target_col']}`")
        st.sidebar.write(f"• **Task:** {st.session_state['problem_type']}")
else:
    st.sidebar.info("No dataset loaded. Upload any CSV file to begin.")

# -----------------------------------------------------------------------------
# PAGE 1: DATASET UPLOAD & OVERVIEW
# -----------------------------------------------------------------------------
if "Dataset Upload" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>Generic Dataset Analysis & ML Platform</h1>
        <p>Upload ANY tabular CSV dataset to automatically audit, preprocess, visualize, and train machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Upload CSV Dataset")
    col_up, col_demo = st.columns([3, 2])
    
    with col_up:
        uploaded_file = st.file_uploader("Choose a CSV file from your computer:", type=['csv'])
        if uploaded_file is not None:
            try:
                new_df = load_csv_data(uploaded_file)
                st.session_state['df'] = new_df
                st.session_state['dataset_name'] = uploaded_file.name
                st.session_state['trained_artifacts'] = None  # Reset training state
                st.session_state['target_col'] = None
                st.session_state['problem_type'] = None
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(new_df)} rows, {new_df.shape[1]} columns)!")
            except Exception as e:
                st.error(f"Error loading uploaded CSV: {e}")
                
    with col_demo:
        st.write("**Or Test with a Sample Dataset:**")
        b1, b2 = st.columns(2)
        if b1.button("🌸 Sample Classification"):
            if os.path.exists(SAMPLE_CLS_PATH):
                st.session_state['df'] = load_csv_data(SAMPLE_CLS_PATH)
                st.session_state['dataset_name'] = "sample_classification.csv"
                st.session_state['target_col'] = "Species"
                st.session_state['problem_type'] = "Classification"
                st.session_state['trained_artifacts'] = None
                st.rerun()
            else:
                st.warning("Sample classification file not found.")
                
        if b2.button("📈 Sample Regression"):
            if os.path.exists(SAMPLE_REG_PATH):
                st.session_state['df'] = load_csv_data(SAMPLE_REG_PATH)
                st.session_state['dataset_name'] = "sample_regression.csv"
                st.session_state['target_col'] = "Disease_Progression"
                st.session_state['problem_type'] = "Regression"
                st.session_state['trained_artifacts'] = None
                st.rerun()
            else:
                st.warning("Sample regression file not found.")

    # Display Dataset Overview if loaded
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        overview = get_dataset_overview(df)
        
        st.markdown("---")
        st.subheader("2. Dataset Summary & Integrity Audit")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Records (Rows)", overview['num_rows'])
        m2.metric("Total Columns", overview['num_cols'])
        m3.metric("Numeric Columns", len(overview['numeric_cols']))
        m4.metric("Categorical Columns", len(overview['categorical_cols']))
        m5.metric("Missing Values", overview['total_missing'])
        
        tab_p, tab_c, tab_s = st.tabs(["📋 Data Preview", "📊 Column Data Types & Statistics", "🔍 Missing & Duplicate Audit"])
        
        with tab_p:
            st.dataframe(df.head(15))
            
        with tab_c:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**Numeric Columns:**")
                st.write(", ".join([f"`{c}`" for c in overview['numeric_cols']]) if overview['numeric_cols'] else "None")
            with col_r:
                st.write("**Categorical Columns:**")
                st.write(", ".join([f"`{c}`" for c in overview['categorical_cols']]) if overview['categorical_cols'] else "None")
                
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe().round(2))
            
        with tab_s:
            st.write(f"• **Duplicate Records Detected:** {overview['duplicate_count']}")
            if overview['total_missing'] > 0:
                missing_df = pd.DataFrame([
                    {'Column': k, 'Missing Count': v, 'Percentage': f"{(v/overview['num_rows'])*100:.2f}%"}
                    for k, v in overview['missing_values'].items() if v > 0
                ])
                st.dataframe(missing_df)
            else:
                st.success("Zero missing values detected across the entire dataset!")
    else:
        st.info("Upload a CSV file above to begin analysis.")

# -----------------------------------------------------------------------------
# PAGE 2: SETUP & MODEL TRAINING
# -----------------------------------------------------------------------------
elif "Setup & Model Training" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>Model Configuration & Training Pipeline</h1>
        <p>Select ANY target column from your uploaded dataset, auto-detect task type, and train models without data leakage.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['df'] is None:
        st.warning("Please upload a dataset first in '1. Dataset Upload & Overview'.")
    else:
        df = st.session_state['df']
        columns = list(df.columns)
        
        st.subheader("1. Target Column Selection")
        col_t, col_type = st.columns(2)
        
        with col_t:
            default_t_idx = columns.index(st.session_state['target_col']) if st.session_state['target_col'] in columns else (len(columns) - 1)
            selected_target = st.selectbox("Select Target Variable (Y):", columns, index=default_t_idx)
            st.session_state['target_col'] = selected_target
            
        # Auto-infer problem type
        inferred_type, reason_str = infer_problem_type(df, selected_target)
        
        with col_type:
            current_choice_idx = 0 if (st.session_state.get('problem_type') == "Classification" or (st.session_state.get('problem_type') is None and inferred_type == "Classification")) else 1
            problem_type_choice = st.radio(
                "Task Type (Auto-Detected with Manual Override):",
                ["Classification", "Regression"],
                index=current_choice_idx,
                help=f"Detection Reason: {reason_str}"
            )
            st.session_state['problem_type'] = problem_type_choice
            st.caption(f"ℹ️ Auto-detection note: {reason_str}")
            
        st.markdown("---")
        st.subheader("2. Feature Selection & Preprocessing Options")
        
        remaining_cols = [c for c in columns if c != selected_target]
        selected_features = st.multiselect(
            "Select Predictor Features (X):",
            remaining_cols,
            default=remaining_cols,
            help="By default, all remaining columns are used. Deselect any IDs or irrelevant columns."
        )
        
        col_split, col_dummy = st.columns([2, 1])
        with col_split:
            test_split = st.slider("Holdout Test Set Size (%):", min_value=10, max_value=40, value=20, step=5) / 100.0
            
        st.info("""
        **Automated DWDM Preprocessing Steps Applied Dynamically:**
        1. Removal of duplicate records.
        2. Imputation of numerical columns with feature median.
        3. Imputation of categorical columns with mode (most frequent).
        4. One-Hot Encoding for categorical features; Label Encoding for classification target.
        5. StandardScaler fitted **strictly on training split** to guarantee no data leakage!
        """)
        
        if st.button("⚡ Run Preprocessing & Train Models", use_container_width=True):
            if len(selected_features) == 0:
                st.error("Please select at least one predictor feature column.")
            else:
                with st.spinner("Executing preprocessing and training all 3 algorithms..."):
                    try:
                        X_train, X_test, y_train, y_test, prep_bundle = preprocess_generic_dataset(
                            df=df,
                            target_col=selected_target,
                            feature_cols=selected_features,
                            problem_type=problem_type_choice,
                            test_size=test_split,
                            random_state=42
                        )
                        
                        if problem_type_choice == 'Classification':
                            metrics, trained_models, cms, best_model, best_score = train_and_evaluate_classifiers(
                                X_train, X_test, y_train, y_test,
                                class_names=prep_bundle['class_names']
                            )
                            st.session_state['trained_artifacts'] = {
                                'problem_type': 'Classification',
                                'target_col': selected_target,
                                'prep_bundle': prep_bundle,
                                'metrics': metrics,
                                'trained_models': trained_models,
                                'confusion_matrices': cms,
                                'best_model': best_model,
                                'best_score': best_score,
                                'X_test': X_test,
                                'y_test': y_test
                            }
                        else:  # Regression
                            metrics, trained_models, test_preds, best_model, best_score = train_and_evaluate_regressors(
                                X_train, X_test, y_train, y_test
                            )
                            st.session_state['trained_artifacts'] = {
                                'problem_type': 'Regression',
                                'target_col': selected_target,
                                'prep_bundle': prep_bundle,
                                'metrics': metrics,
                                'trained_models': trained_models,
                                'test_predictions': test_preds,
                                'best_model': best_model,
                                'best_score': best_score,
                                'X_test': X_test,
                                'y_test': y_test
                            }
                            
                        st.success(f"Training completed successfully! Optimal Model: **{best_model}**")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error during training pipeline: {str(e)}")
                        
        if st.session_state['trained_artifacts'] is not None:
            st.success(f"Models are active and cached! Current Best Algorithm: **{st.session_state['trained_artifacts']['best_model']}**")

# -----------------------------------------------------------------------------
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
elif "Exploratory Data Analysis" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>Exploratory Data Analysis (EDA)</h1>
        <p>Interactive statistical visualizations dynamically rendered from your uploaded dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['df'] is None:
        st.warning("Please upload a dataset first in '1. Dataset Upload & Overview'.")
    else:
        df = st.session_state['df']
        overview = get_dataset_overview(df)
        target = st.session_state['target_col'] if st.session_state['target_col'] in df.columns else df.columns[-1]
        prob_type = st.session_state['problem_type'] if st.session_state['problem_type'] else infer_problem_type(df, target)[0]
        
        tab_tgt, tab_corr, tab_miss, tab_bivar = st.tabs([
            "🎯 Target Variable Distribution",
            "🔥 Pearson Correlation Heatmap",
            "🔍 Missing Values Analysis",
            "📈 Dynamic Bivariate Plots"
        ])
        
        with tab_tgt:
            st.subheader(f"Target Distribution: '{target}' ({prob_type})")
            fig_tgt = plot_target_distribution(df, target_col=target, problem_type=prob_type)
            if fig_tgt:
                st.pyplot(fig_tgt)
                plt.close(fig_tgt)
            else:
                st.info("Unable to plot target distribution.")
                
        with tab_corr:
            st.subheader("Pearson Correlation Heatmap")
            if len(overview['numeric_cols']) >= 2:
                fig_corr = plot_correlation_matrix(df, overview['numeric_cols'])
                if fig_corr:
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)
            else:
                st.info("Not enough numeric columns (at least 2 required) to compute correlation.")
                
        with tab_miss:
            st.subheader("Missing Values Count")
            fig_miss = plot_missing_values(df)
            if fig_miss:
                st.pyplot(fig_miss)
                plt.close(fig_miss)
            else:
                st.success("Zero missing values in this dataset!")
                
        with tab_bivar:
            st.subheader("Dynamic Feature vs. Target Exploration")
            available_features = [c for c in df.columns if c != target]
            if available_features:
                chosen_feat = st.selectbox("Select Feature Column to Explore against Target:", available_features)
                fig_bi = plot_feature_vs_target(df, feature_col=chosen_feat, target_col=target, problem_type=prob_type)
                if fig_bi:
                    st.pyplot(fig_bi)
                    plt.close(fig_bi)
            else:
                st.info("No feature columns available.")

# -----------------------------------------------------------------------------
# PAGE 4: ALGORITHM BENCHMARKING
# -----------------------------------------------------------------------------
elif "Algorithm Benchmarking" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>Algorithm Benchmarking & Comparison</h1>
        <p>Comparative metrics, grouped bar charts, and diagnostic plots across 3 standard data mining algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['trained_artifacts'] is None:
        st.warning("Please configure and train models in '2. Setup & Model Training' first.")
    else:
        artifacts = st.session_state['trained_artifacts']
        prob_type = artifacts['problem_type']
        metrics = artifacts['metrics']
        best_model = artifacts['best_model']
        best_score = artifacts['best_score']
        
        # Best Model Banner
        metric_highlight = f"Test Accuracy: {best_score}%" if prob_type == 'Classification' else f"R² Score: {best_score}"
        st.markdown(f"""
        <div class="best-model-banner">
            <div>
                <span style="font-size: 1.05rem; opacity: 0.9;">🏆 Identified Optimal Model</span>
                <h2 style="margin: 0; color: white;">Best Model: {best_model}</h2>
            </div>
            <div style="text-align: right;">
                <h2 style="margin: 0; color: white;">{metric_highlight}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Table
        st.subheader("📊 Comparative Evaluation Table")
        df_comp = pd.DataFrame(metrics).T
        df_comp.index.name = "Algorithm"
        
        if prob_type == 'Classification':
            st.dataframe(df_comp.style.highlight_max(axis=0, color="#bbf7d0").format("{:.2f}%"))
            
            st.subheader("📈 Algorithm Comparison Graph")
            fig_comp = plot_classification_comparison(metrics)
            st.pyplot(fig_comp)
            plt.close(fig_comp)
            
            st.subheader("🔍 Confusion Matrix Diagnostics")
            cm_cols = st.columns(len(artifacts['confusion_matrices']))
            class_names = artifacts['prep_bundle']['class_names']
            for idx, (name, cm_arr) in enumerate(artifacts['confusion_matrices'].items()):
                with cm_cols[idx]:
                    fig_cm = plot_confusion_matrix(np.array(cm_arr), class_names, title=name)
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                    
        else:  # Regression
            st.dataframe(df_comp.style.highlight_max(subset=['R2_Score'], color="#bbf7d0").highlight_min(subset=['RMSE', 'MAE', 'MSE'], color="#fed7aa"))
            
            st.subheader("📈 Algorithm Comparison Graphs")
            fig_comp = plot_regression_comparison(metrics)
            st.pyplot(fig_comp)
            plt.close(fig_comp)
            
            st.subheader("🔍 Actual vs. Predicted Residual Diagnostics")
            pred_cols = st.columns(len(artifacts['test_predictions']))
            for idx, (name, pred_data) in enumerate(artifacts['test_predictions'].items()):
                with pred_cols[idx]:
                    fig_res = plot_regression_actual_vs_predicted(
                        pred_data['actual'], pred_data['predicted'], title=name
                    )
                    st.pyplot(fig_res)
                    plt.close(fig_res)

# -----------------------------------------------------------------------------
# PAGE 5: LIVE PREDICTION PORTAL
# -----------------------------------------------------------------------------
elif "Live Prediction Portal" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>Dynamic Prediction Portal</h1>
        <p>Interactive inference engine with dynamically generated input controls for your trained feature set.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['trained_artifacts'] is None:
        st.warning("Please configure and train models in '2. Setup & Model Training' first.")
    else:
        artifacts = st.session_state['trained_artifacts']
        prob_type = artifacts['problem_type']
        target_col = artifacts['target_col']
        prep_bundle = artifacts['prep_bundle']
        trained_models = artifacts['trained_models']
        best_model_name = artifacts['best_model']
        original_features = prep_bundle['original_features']
        df = st.session_state['df']
        
        st.subheader("1. Enter Feature Values for Inference")
        
        # Dynamically build input form
        with st.form("dynamic_prediction_form"):
            input_values = {}
            
            cols = st.columns(3)
            for i, feat in enumerate(original_features):
                col_widget = cols[i % 3]
                with col_widget:
                    is_numeric = feat in prep_bundle['num_cols']
                    
                    if is_numeric:
                        series = pd.to_numeric(df[feat], errors='coerce').dropna()
                        min_v = float(series.min()) if len(series) > 0 else 0.0
                        max_v = float(series.max()) if len(series) > 0 else 100.0
                        med_v = float(series.median()) if len(series) > 0 else (min_v + max_v) / 2.0
                        step_v = 1.0 if (series % 1 == 0).all() else 0.1
                        
                        input_values[feat] = st.number_input(
                            f"{feat}",
                            min_value=min_v,
                            max_value=max_v,
                            value=med_v,
                            step=step_v
                        )
                    else:
                        categories = list(df[feat].dropna().unique())
                        if len(categories) == 0:
                            categories = ["Standard"]
                        categories_str = [str(c) for c in categories]
                        input_values[feat] = st.selectbox(f"{feat}", categories_str, index=0)
                        
            st.markdown("---")
            col_m, col_b = st.columns([2, 1])
            with col_m:
                model_choices = [f"Best Model ({best_model_name})"] + list(trained_models.keys())
                selected_model_choice = st.selectbox("Select Model for Inference:", model_choices)
                
                chosen_model_obj = trained_models[best_model_name] if "Best Model" in selected_model_choice else trained_models[selected_model_choice]
                active_model_name = best_model_name if "Best Model" in selected_model_choice else selected_model_choice
                
            with col_b:
                st.write("")
                st.write("")
                submit_pred = st.form_submit_button("🔮 Predict Target Value", use_container_width=True)
                
        # Inference Execution
        if submit_pred:
            with st.spinner("Processing input and calculating prediction..."):
                try:
                    pred_res = predict_sample(
                        input_dict=input_values,
                        model=chosen_model_obj,
                        preprocessor_bundle=prep_bundle,
                        model_name=active_model_name
                    )
                    
                    if prob_type == 'Classification':
                        st.markdown(f"""
                        <div class="pred-card">
                            <h4 style="margin: 0; text-transform: uppercase;">Predicted '{target_col}'</h4>
                            <h1 style="font-size: 2.8rem; margin: 8px 0;">{str(pred_res['prediction']).upper()}</h1>
                            <p style="margin: 0; font-size: 1.1rem;">Model: <b>{active_model_name}</b> | Confidence: <b>{pred_res['confidence']}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("📊 Class Confidence Breakdown")
                        for cls_lbl, prob in pred_res['probabilities'].items():
                            st.write(f"**{cls_lbl}**: {prob}%")
                            st.progress(min(prob / 100.0, 1.0))
                            
                    else:  # Regression
                        pred_val = pred_res['prediction']
                        dataset_mean = float(pd.to_numeric(df[target_col], errors='coerce').mean())
                        
                        st.markdown(f"""
                        <div class="pred-card">
                            <h4 style="margin: 0; text-transform: uppercase;">Predicted '{target_col}' Value</h4>
                            <h1 style="font-size: 2.8rem; margin: 8px 0;">{pred_val}</h1>
                            <p style="margin: 0; font-size: 1.1rem;">Model: <b>{active_model_name}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Predicted Value", pred_val)
                        c2.metric("Dataset Average Baseline", round(dataset_mean, 2), delta=round(pred_val - dataset_mean, 2))
                        
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")

# -----------------------------------------------------------------------------
# PAGE 6: ABOUT PLATFORM
# -----------------------------------------------------------------------------
elif "About Platform" in nav_choice:
    st.markdown("""
    <div class="main-banner">
        <h1>About the Generic DWDM Platform</h1>
        <p>A versatile Data Warehousing and Data Mining (DWDM) application supporting any tabular CSV dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Platform Objectives")
    st.write("""
    This platform transforms specialized academic machine learning code into a **flexible, general-purpose 
    data mining and predictive modeling workbench**. Rather than being restricted to a single hard-coded dataset, 
    the system dynamically ingests ANY structured CSV file, automatically manages data cleaning, provides 
    exploratory data analysis, trains and benchmarks standard algorithms, and delivers instant predictions.
    """)
    
    st.subheader("2. Key Data Mining (DWDM) Concepts Implemented")
    st.markdown("""
    - **Dynamic Ingestion & Schema Discovery**: Delimiter sniffing, automatic type inference (numeric vs. categorical).
    - **Missing Value Handling**: Robust imputation using feature medians for continuous attributes and modes for categorical attributes.
    - **Duplicate Pruning**: Identifies and removes duplicate records to prevent sample bias.
    - **Categorical Feature Encoding**: One-Hot Encoding (`OneHotEncoder`) ensures clean numeric conversion without arbitrary ordinal bias.
    - **Strict No-Data-Leakage Guarantee**: All transformers (imputers, encoders, and `StandardScaler`) are fitted strictly on the training partition ($X_{train}$) and transformed onto test and query vectors.
    - **Dual Task Support**: Automatically distinguishes between **Classification** (discrete target) and **Regression** (continuous target).
    """)
    
    st.subheader("3. Algorithms Implemented")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Classification Algorithms:**
        1. **Decision Tree Classifier (CART)**: Gini impurity recursive tree splits.
        2. **Random Forest Classifier**: Ensemble bagging of 100 decorrelated decision trees.
        3. **K-Nearest Neighbors (KNN)**: Distance-weighted majority voting in standardized Euclidean space.
        """)
    with col2:
        st.markdown("""
        **Regression Algorithms:**
        1. **Decision Tree Regressor**: Variance reduction splits.
        2. **Random Forest Regressor**: Ensemble averaging across 100 regression trees.
        3. **KNN Regressor**: Inverse distance-weighted continuous neighbor averaging.
        """)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "BTech CSE DWDM Mini Project | Generic Dataset Analysis and Machine Learning Platform"
    "</div>",
    unsafe_allow_html=True
)
