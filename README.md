# Generic Dataset Analysis and Machine Learning Platform
**An End-to-End Data Warehousing & Data Mining (DWDM) Application**  
*Department of Computer Science & Engineering | Academic Year 2026*

---

## 📌 1. Project Title & Overview
**"Generic Dataset Analysis and Machine Learning Platform Using Data Mining Algorithms"**

This application is a flexible, general-purpose **Data Mining & Predictive Modeling Workbench**. Rather than being restricted to a single fixed dataset or hard-coded feature names, the platform allows users to **upload ANY tabular CSV dataset** via the Streamlit interface. It automatically detects attribute types, performs leakage-free preprocessing, discovers patterns via Exploratory Data Analysis (EDA), benchmarks algorithms, and provides real-time dynamic predictions.

---

## 🚀 2. Core Objectives & DWDM Concepts
- **Dynamic Dataset Ingestion**: Ingests arbitrary CSV files with automated delimiter and schema detection.
- **Automated Data Cleaning (DWDM)**:
  - **Deduplication**: Identifies and removes duplicate records.
  - **Imputation**: Median imputation for numerical features; mode (most frequent) imputation for categorical features.
  - **Zero-Variance Pruning**: Automatically detects and drops uninformative constant columns.
- **Categorical Feature Encoding**: Uses `OneHotEncoder` on feature columns to avoid arbitrary ordinal bias.
- **Strict No-Data-Leakage Guarantee**: Transformers (imputers, encoders, and `StandardScaler`) are fitted strictly on $X_{train}$ and transformed onto $X_{test}$ and incoming prediction vectors.
- **Dual Problem Support**: Automatically infers whether the target variable represents **Classification** (discrete classes) or **Regression** (continuous numbers) with manual override.
- **Algorithm Benchmarking**: Compares three fundamental data mining algorithms (Decision Tree, Random Forest, KNN) for both classification and regression tasks.
- **Interactive Decision Support**: Automatically renders dynamic sliders and dropdowns for any feature set to execute live inference with confidence breakdowns.

---

## 🛠️ 3. Technologies Used
- **Programming Language**: Python 3.13+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Data Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Serialization**: Joblib

---

## 📁 4. Project Directory Structure
```text
studentperformance/
│
├── data/
│   ├── sample_classification.csv      # Sample 1: Multi-class Classification Dataset
│   └── sample_regression.csv          # Sample 2: Continuous Regression Dataset
│
├── models/                            # Serialized model and preprocessor cache
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_loader.py                 # Dynamic CSV loader, schema auditor, problem type inference
│   ├── preprocessing.py               # Leakage-free preprocessing, imputation, One-Hot Encoding, scaling
│   ├── eda.py                         # Dynamic EDA plots (target distributions, heatmaps, residuals)
│   ├── classification.py              # Decision Tree, Random Forest, KNN classifiers & metrics
│   ├── regression.py                  # Decision Tree, Random Forest, KNN regressors & metrics
│   └── prediction.py                  # Dynamic live inference engine for any feature inputs
│
├── .streamlit/
│   └── config.toml                    # Streamlit headless server configuration
├── app.py                             # Modern Streamlit Web Application
├── test_generic_platform.py           # Automated test suite for classification, regression, and custom CSVs
├── requirements.txt                   # Project dependencies
└── README.md                          # Full DWDM project documentation
```

---

## 🔄 5. Platform Workflow

```text
[ Upload Any CSV ]  or  [ Load Benchmark Dataset ]
        │
        ▼
[ Automated Schema Audit ] ──► (Rows, Cols, Numeric/Categorical Types, Missing Values, Duplicates)
        │
        ▼
[ Select Target Column ] ──► (Auto-detects Classification vs. Regression)
        │
        ▼
[ Data Preprocessing Pipeline ]
  • Remove duplicates
  • Impute missing values (median/mode)
  • One-Hot Encode categorical features
  • Train-Test Partition (Stratified for classification)
  • StandardScaler fitted strictly on Train split (No Data Leakage)
        │
        ▼
[ Machine Learning Benchmarking ]
  • Classification: Decision Tree vs. Random Forest vs. KNN
  • Regression: Decision Tree Regressor vs. Random Forest Regressor vs. KNN Regressor
        │
        ▼
[ Model Evaluation & Comparison ]
  • Comparative Metrics Table & Best Model Banner
  • Grouped Bar Charts
  • Diagnostics: Confusion Matrices (Classification) / Actual vs. Predicted (Regression)
        │
        ▼
[ Live Dynamic Prediction Portal ]
  • Auto-generates input controls for all features
  • Real-time prediction with probability confidence
```

---

## 🤖 6. Algorithms & Evaluation Metrics

### A. Classification Algorithms & Metrics
- **Algorithms**:
  1. **Decision Tree (CART)**: Gini impurity recursive tree splits.
  2. **Random Forest Classifier**: Ensemble bagging across 100 decorrelated decision trees.
  3. **K-Nearest Neighbors (KNN)**: Distance-weighted majority voting in standardized Euclidean space.
- **Metrics**:
  - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  - $\text{Weighted Precision} = \sum \frac{TP_i}{TP_i + FP_i} \times w_i$
  - $\text{Weighted Recall} = \sum \frac{TP_i}{TP_i + FN_i} \times w_i$
  - $\text{Weighted F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  - Multi-class Confusion Matrix Heatmaps.

### B. Regression Algorithms & Metrics
- **Algorithms**:
  1. **Decision Tree Regressor**: Variance-reduction recursive partitioning.
  2. **Random Forest Regressor**: Ensemble averaging across 100 regression trees.
  3. **KNN Regressor**: Inverse distance-weighted continuous neighbor averaging.
- **Metrics**:
  - $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$
  - $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
  - $\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
  - $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
  - Actual vs. Predicted Scatter Plots.

---

## 📊 7. Example Test Datasets Included

The platform is designed to process **ANY user-uploaded CSV**. For quick testing, two standard sample datasets are included:

### Sample 1: Classification Task
- **Dataset**: `data/sample_classification.csv` (150 records, 4 features)
- **Target Variable**: `Species` (Setosa, Versicolor, Virginica)
- **Results**:
  - **K-Nearest Neighbors** 🏆: **100.0% Accuracy** | **100.0% F1-Score** (Best Model)
  - **Random Forest**: **96.67% Accuracy** | **96.66% F1-Score**
  - **Decision Tree**: **96.67% Accuracy** | **96.66% F1-Score**

### Sample 2: Regression Task
- **Dataset**: `data/sample_regression.csv` (442 records, 10 continuous features)
- **Target Variable**: `Disease_Progression` (Continuous metric)
- **Results**:
  - **KNN Regressor** 🏆: **$R^2 = 0.445$** | **$\text{RMSE} = 54.25$** (Best Model)
  - **Random Forest Regressor**: **$R^2 = 0.443$** | **$\text{RMSE} = 54.31$**
  - **Decision Tree Regressor**: **$R^2 = 0.144$** | **$\text{RMSE} = 67.35$**

---

## 💻 8. How to Install and Run

### Step 1: Navigate to Project Directory
```bash
cd /Users/hemanth/Desktop/studentperformance
```

### Step 2: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 3: Launch Streamlit Web Application
```bash
streamlit run app.py
```
*Access in browser at: **`http://localhost:8501`***

### Step 4: Run Automated Integrity Test Suite
```bash
python test_generic_platform.py
```

---

## 🛡️ 9. Robustness & Edge-Case Handling
The platform handles real-world data issues:
- **Missing Values**: Handled gracefully using median/mode imputers without crashing.
- **Mixed Data Types**: Automatically segregates numeric and categorical columns.
- **High Cardinality**: Categorical columns are transformed using `OneHotEncoder(handle_unknown='ignore')`.
- **Constant Columns**: Automatically pruned before scaling.
- **Target Integrity**: Checks that classification targets contain at least 2 distinct classes and regression targets are numeric.
