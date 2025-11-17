# Diabetes Dataset Preprocessing

## Overview

This project provides a complete data preprocessing pipeline for the Pima Indians Diabetes Database. The pipeline includes data loading, exploratory data analysis (EDA), outlier handling, missing value imputation, feature normalization, and train-validation splitting. The project consists of an exploratory Jupyter notebook for analysis and an automated Python script for production-ready preprocessing.

**Dataset:** Pima Indians Diabetes Database
**Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Key Features

- Automated preprocessing pipeline with modular functions
- Outlier detection and handling using IQR method
- Missing value imputation with median strategy
- Feature normalization using StandardScaler
- Train-validation split with configurable ratio
- Saved preprocessing artifacts (scaler, preprocessed data)
- Interactive Jupyter notebook for data exploration
- Reusable functions for loading preprocessed data

### Project Components

1. **automate_Lukas.py**: Production-ready automated preprocessing script
2. **Eksperimen_Lukas.ipynb**: Interactive notebook for EDA and experimentation
3. **diabetes_preprocessing/**: Output directory containing preprocessed data

## Dataset

This project uses the **Pima Indians Diabetes Database** from Kaggle.

**Dataset Source:**
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### About the Dataset

The Pima Indians Diabetes Database is a well-known dataset originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contains diagnostic measurements for predicting whether a patient has diabetes.

**Dataset Characteristics:**
- **Instances**: 768 patients
- **Features**: 8 numerical features
- **Target**: Binary classification (0 = No diabetes, 1 = Diabetes)
- **Population**: Females of Pima Indian heritage, at least 21 years old

**Features:**

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg / height in m^2)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)
8. **Age**: Age in years

**Target Variable:**
- **Outcome**: 0 (No diabetes) or 1 (Diabetes)

### Dataset Citation

```
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. 
In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). 
IEEE Computer Society Press.
```

## Project Structure

```
preprocessing/
├── automate_Lukas.py              # Automated preprocessing script
├── Eksperimen_Lukas.ipynb         # Exploratory data analysis notebook
├── diabetes_preprocessing/        # Output directory
│   ├── diabetes_preprocessed.csv  # Complete preprocessed dataset
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   ├── X_train.csv                # Training features
│   ├── X_val.csv                  # Validation features
│   ├── y_train.csv                # Training labels
│   └── y_val.csv                  # Validation labels
├── diabetes.csv                   # Raw dataset (to be downloaded)
└── requirements.txt               # Python dependencies
```

## Requirements

### System Requirements

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for interactive exploration)
- Minimum 2GB RAM
- 500MB free disk space

### Python Dependencies

All required dependencies are listed in `requirements.txt`:

```
pandas==2.3.3
numpy==2.3.0
scikit-learn==1.6.0
matplotlib==3.9.2
seaborn==0.13.2
```

## Installation

### 1. Clone or Navigate to Project Directory

```bash
cd preprocessing
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset

Download the Pima Indians Diabetes Database from Kaggle:

**Option 1: Manual Download**
1. Visit https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place the file in the `preprocessing/` directory

**Option 2: Using Kaggle API**
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle API credentials (place kaggle.json in ~/.kaggle/)
# Download dataset
kaggle datasets download -d uciml/pima-indians-diabetes-database
unzip pima-indians-diabetes-database.zip
```

### 5. Verify Installation

```bash
python -c "import numpy, pandas, sklearn; print('Installation successful')"
```

## Usage

### Option 1: Automated Preprocessing (Recommended for Production)

Run the automated preprocessing script to generate all preprocessed data:

```bash
python automate_Lukas.py
```

**What it does:**

1. Loads `diabetes.csv` from the parent directory
2. Handles outliers in Insulin and DiabetesPedigreeFunction using IQR method
3. Handles missing values (zeros treated as NaN, filled with median)
4. Normalizes all features using StandardScaler
5. Splits data into 70% training and 30% validation sets
6. Saves all outputs to `diabetes_preprocessing/` directory:
   - `X_train.csv`: Training features (normalized)
   - `X_val.csv`: Validation features (normalized)
   - `y_train.csv`: Training labels
   - `y_val.csv`: Validation labels
   - `diabetes_preprocessed.csv`: Complete preprocessed dataset
   - `scaler.pkl`: Fitted StandardScaler for future use

**Output Example:**
```
Dataset loaded successfully. Shape: (768, 9)
Outliers handled for column: Insulin
Outliers handled for column: DiabetesPedigreeFunction
Missing values handled successfully
Feature normalization completed
Features shape: (768, 8), Target shape: (768,)
Data split completed:
Training set: (537, 8), Validation set: (231, 8)
Scaler saved to 'diabetes_preprocessing/scaler.pkl'
Preprocessed data saved to 'diabetes_preprocessing/' folder
Preprocessing pipeline completed successfully!
```

### Option 2: Interactive Exploration

For exploratory data analysis and experimentation:

```bash
jupyter notebook Eksperimen_Lukas.ipynb
```

**Notebook Contents:**

1. **Dataset Introduction**: Overview and source information
2. **Import Libraries**: Required Python packages
3. **Load Dataset**: Reading the CSV file
4. **Exploratory Data Analysis (EDA)**:
   - Data inspection (shape, types, info)
   - Statistical summaries
   - Missing value analysis
   - Distribution analysis
   - Correlation analysis
   - Visualizations

The notebook provides an interactive environment to understand the data before applying automated preprocessing.

### Option 3: Using Preprocessing Functions in Your Code

You can import and use the preprocessing functions in your own scripts:

```python
from automate_Lukas import preprocess_diabetes_data, load_preprocessed_data

# Run preprocessing pipeline
X_train, X_val, y_train, y_val, scaler = preprocess_diabetes_data(
    file_path='diabetes.csv',
    test_size=0.3,
    random_state=42,
    save_scaler=True
)

# Later, load preprocessed data
X_train, X_val, y_train, y_val, scaler = load_preprocessed_data()
```

## Preprocessing Pipeline

The automated preprocessing pipeline follows these steps:

### 1. Data Loading

Loads the diabetes dataset from CSV file and validates the file path.

### 2. Outlier Handling

Uses the Interquartile Range (IQR) method to handle outliers:

- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Computes IQR = Q3 - Q1
- Defines bounds: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Clips values outside bounds to the boundary values
- Applied to: **Insulin** and **DiabetesPedigreeFunction**

**Formula:**
```
Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

### 3. Missing Value Handling

Handles missing values represented as zeros in medical features:

- Columns affected: **Glucose**, **BloodPressure**, **SkinThickness**, **Insulin**, **BMI**
- Strategy: Replace 0 with NaN, then fill with median of the column
- Rationale: Medical measurements cannot be zero (physiologically impossible)

### 4. Feature Normalization

Applies StandardScaler to normalize all features:

- Method: Z-score normalization
- Formula: `z = (x - μ) / σ`
  - μ (mu): mean of the feature
  - σ (sigma): standard deviation of the feature
- Result: Features have mean=0 and standard deviation=1
- Applied to all 8 features
- Scaler is saved for future use on new data

### 5. Train-Validation Split

Splits data into training and validation sets:

- Default ratio: 70% training, 30% validation
- Random state: 42 (for reproducibility)
- Stratification: Not applied (can be added if needed)

### 6. Data Export

Saves all preprocessed data and artifacts:

- CSV files for easy inspection and use
- Pickle file for the scaler object
- All files stored in `diabetes_preprocessing/` directory

## Available Functions

### Core Functions

#### `load_data(file_path='diabetes.csv')`
Loads the dataset from CSV file.

**Parameters:**
- `file_path` (str): Path to the dataset file

**Returns:**
- `pd.DataFrame`: Loaded dataset

#### `handle_outliers(df, columns=['Insulin', 'DiabetesPedigreeFunction'])`
Handles outliers using IQR method.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `columns` (list): Columns to process

**Returns:**
- `pd.DataFrame`: Dataframe with outliers handled

#### `handle_missing_values(df, cols_invalid_zero=[...])`
Handles missing values represented as zeros.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `cols_invalid_zero` (list): Columns where 0 is invalid

**Returns:**
- `pd.DataFrame`: Dataframe with missing values handled

#### `normalize_features(df, num_cols=[...])`
Normalizes features using StandardScaler.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `num_cols` (list): Columns to normalize

**Returns:**
- `tuple`: (normalized_dataframe, fitted_scaler)

#### `prepare_features_target(df, target_column='Outcome')`
Separates features and target variable.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `target_column` (str): Name of target column

**Returns:**
- `tuple`: (features_dataframe, target_series)

### Pipeline Functions

#### `preprocess_diabetes_data(file_path='diabetes.csv', test_size=0.3, random_state=42, save_scaler=True)`
Complete end-to-end preprocessing pipeline.

**Parameters:**
- `file_path` (str): Path to dataset
- `test_size` (float): Proportion for validation set
- `random_state` (int): Random seed for reproducibility
- `save_scaler` (bool): Whether to save the fitted scaler

**Returns:**
- `tuple`: (X_train, X_val, y_train, y_val, scaler)

#### `load_preprocessed_data(data_dir=None)`
Loads previously preprocessed data.

**Parameters:**
- `data_dir` (str, optional): Directory containing preprocessed files

**Returns:**
- `tuple`: (X_train, X_val, y_train, y_val, scaler)

## Customization

### Changing Train-Validation Split Ratio

Edit the `test_size` parameter in `automate_Lukas.py`:

```python
X_train, X_val, y_train, y_val, scaler = preprocess_diabetes_data(
    file_path='diabetes.csv',
    test_size=0.2,  # 80-20 split
    random_state=42
)
```

### Adding More Outlier Columns

Modify the `columns` parameter in the `handle_outliers` function call:

```python
df = handle_outliers(df, columns=['Insulin', 'DiabetesPedigreeFunction', 'Age'])
```

### Changing Missing Value Strategy

Modify the `handle_missing_values` function to use different imputation:

```python
# Use mean instead of median
df_processed[cols_invalid_zero] = df_processed[cols_invalid_zero].fillna(
    df_processed[cols_invalid_zero].mean()
)
```

### Using Different Scaler

Replace StandardScaler with other scalers:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # Instead of StandardScaler
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
```

## Output Files

### diabetes_preprocessed.csv

Complete preprocessed dataset with all features normalized and target variable.

**Format:**
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
-0.844,-0.876,-1.024,-1.264,-1.259,-1.245,-0.696,-0.956,1
```

### X_train.csv, X_val.csv

Training and validation features (normalized).

**Shape:**
- X_train: (537, 8)
- X_val: (231, 8)

### y_train.csv, y_val.csv

Training and validation labels (0 or 1).

**Shape:**
- y_train: (537,)
- y_val: (231,)

### scaler.pkl

Fitted StandardScaler object saved using pickle. Can be loaded and used to transform new data:

```python
import pickle

with open('diabetes_preprocessing/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Transform new data
new_data_normalized = scaler.transform(new_data)
```

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- scikit-learn Documentation: https://scikit-learn.org/stable/
- Pandas Documentation: https://pandas.pydata.org/docs/
- IQR Outlier Detection: https://en.wikipedia.org/wiki/Interquartile_range

