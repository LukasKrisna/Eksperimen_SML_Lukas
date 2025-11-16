# Eksperimen_SML_Lukas

This repository contains the machine learning experimentation project for the diabetes dataset, including automated preprocessing pipeline and GitHub Actions workflow.

## Project Structure

```
Eksperimen_SML_Lukas/
├── .github/
│   └── workflows/
│       └── preprocess.yml          # GitHub Actions workflow for automated preprocessing
├── diabetes.csv                    # Raw diabetes dataset
├── preprocessing/
│   ├── Eksperimen_Lukas.ipynb     # Jupyter notebook with EDA and experimentation
│   ├── automate_Lukas.py          # Automated preprocessing script
│   └── diabetes_preprocessing/     # Output folder for preprocessed data
│       ├── X_train.csv            # Training features
│       ├── X_val.csv              # Validation features
│       ├── y_train.csv            # Training labels
│       ├── y_val.csv              # Validation labels
│       ├── diabetes_preprocessed.csv  # Complete preprocessed dataset
│       └── scaler.pkl             # Fitted scaler for normalization
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Features

### 📊 Data Exploration (Basic - 2 pts)

- ✅ Data loading from CSV file
- ✅ Comprehensive Exploratory Data Analysis (EDA)
- ✅ Data preprocessing steps including:
  - Outlier detection and handling using IQR method
  - Missing value imputation with median values
  - Feature normalization using QuantileTransformer
  - Data splitting for training and validation

### 🔧 Automated Preprocessing (Skilled - 3 pts)

- ✅ `automate_Lukas.py` script with modular functions:
  - `load_data()`: Load dataset from CSV
  - `handle_outliers()`: Remove outliers using IQR method
  - `handle_missing_values()`: Handle missing/invalid values
  - `normalize_features()`: Normalize features using QuantileTransformer
  - `prepare_features_target()`: Separate features and target variable
  - `preprocess_diabetes_data()`: Complete preprocessing pipeline
  - `load_preprocessed_data()`: Load previously processed data

### 🚀 GitHub Actions Workflow (Advance - 4 pts)

- ✅ Automated preprocessing trigger on:
  - Push to main/master branch
  - Changes to dataset or preprocessing script
  - Manual workflow dispatch
- ✅ Features:
  - Automatic dependency installation
  - Data preprocessing execution
  - Artifact upload for processed data
  - Automatic commit of processed data
  - Data validation and preview

## Usage

### Manual Preprocessing

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing script
cd preprocessing
python automate_Lukas.py
```

### Automated Preprocessing via GitHub Actions

1. Push changes to the repository
2. GitHub Actions will automatically trigger the preprocessing workflow
3. Processed data will be available as artifacts and committed to the repository

## Data Processing Steps

1. **Data Loading**: Load the diabetes dataset from CSV file
2. **Outlier Handling**: Use IQR method to cap outliers in Insulin and DiabetesPedigreeFunction columns
3. **Missing Value Treatment**: Replace invalid zero values with NaN and impute with median values
4. **Feature Normalization**: Apply QuantileTransformer with normal distribution output
5. **Data Splitting**: Split into training (70%) and validation (30%) sets
6. **Artifact Generation**: Save processed data and fitted scaler for reuse

## Dataset Information

The diabetes dataset contains 768 instances with 8 features:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1) - Target variable

## Dependencies

- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2

## Author

**Lukas** - Machine Learning System Development Project
