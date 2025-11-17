# Diabetes Classification Model Building

## Overview

This project implements a comprehensive machine learning pipeline for diabetes classification using the Pima Indians Diabetes Database from Kaggle. The project includes three training approaches: basic model training, hyperparameter tuning with SMOTE, and multi-model comparison. All experiments are tracked using MLflow with optional DagsHub integration for remote experiment management.

**Dataset:** Pima Indians Diabetes Database
**Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Key Features

- Multiple machine learning algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Class imbalance handling using SMOTE (Synthetic Minority Over-sampling Technique)
- Comprehensive experiment tracking with MLflow
- DagsHub integration for remote experiment tracking
- Automated visualization generation (confusion matrix, ROC curves, feature importance)
- Detailed performance metrics logging
- Model comparison and selection

### Training Approaches

1. **Basic Model** (`modelling.py`): Baseline Random Forest classifier without hyperparameter tuning
2. **Advanced Model** (`modelling_tuning.py`): Random Forest with GridSearchCV tuning and SMOTE
3. **Model Comparison** (`modelling_comparison.py`): Comparative analysis of three different algorithms

### Technology Stack

- **Machine Learning**: scikit-learn, imbalanced-learn
- **Experiment Tracking**: MLflow 2.19.0
- **Remote Tracking**: DagsHub
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Python Version**: 3.12.7

## Project Structure

```
Membangun_model/
├── diabetes_preprocessing/
│   ├── diabetes_preprocessed.csv  # Preprocessed dataset
│   ├── scaler.pkl                 # Feature scaler
│   ├── X_train.csv                # Training features
│   ├── X_val.csv                  # Validation features
│   ├── y_train.csv                # Training labels
│   └── y_val.csv                  # Validation labels
├── mlruns/                        # Local MLflow tracking directory
├── screenshot/
│   ├── dagshub/                   # DagsHub experiment screenshots
│   └── local/                     # Local MLflow UI screenshots
│       ├── base_model/
│       └── tuning_model/
├── modelling.py                   # Basic model training script
├── modelling_tuning.py            # Advanced model with tuning
├── modelling_comparison.py        # Multi-model comparison
├── requirements.txt               # Python dependencies
├── DagsHub.txt                    # DagsHub repository links
└── .gitignore                     # Git ignore patterns
```

## Requirements

### System Requirements

- Python 3.12.7 or higher
- Minimum 4GB RAM
- 2GB free disk space

### Python Dependencies

All required dependencies are listed in `requirements.txt`:

```
pandas==2.3.3
numpy==2.3.0
scikit-learn==1.5.2
matplotlib==3.10
seaborn==0.13
mlflow==2.19.0
imbalanced-learn==0.12.3
```

## Installation

### 1. Clone or Navigate to Project Directory

```bash
cd Membangun_model
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

### 4. Verify Installation

```bash
python -c "import mlflow, sklearn, pandas; print('Installation successful')"
```

## Configuration

### Local MLflow Tracking (Default)

No additional configuration needed. Experiments will be tracked locally in the `mlruns/` directory.

### DagsHub Remote Tracking (Optional)

To track experiments on DagsHub, set the following environment variables:

```bash
export MLFLOW_TRACKING_URI='https://dagshub.com/lukaskrisnaaa/mlsystem-submission.mlflow'
export MLFLOW_TRACKING_USERNAME='your-dagshub-username'
export MLFLOW_TRACKING_PASSWORD='your-dagshub-token'
```

**Windows:**
```cmd
set MLFLOW_TRACKING_URI=https://dagshub.com/lukaskrisnaaa/mlsystem-submission.mlflow
set MLFLOW_TRACKING_USERNAME=your-dagshub-username
set MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

**Get DagsHub Token:**
1. Sign up at https://dagshub.com
2. Go to Settings > Tokens
3. Generate a new token

## Usage

### 1. Basic Model Training

Train a baseline Random Forest model without hyperparameter tuning:

```bash
python modelling.py
```

**What it does:**
- Loads preprocessed diabetes data
- Trains Random Forest with default parameters
- Evaluates on validation set
- Logs metrics and model to MLflow
- Experiment name: `Diabetes_Classification_Basic`

**Output metrics:**
- Training accuracy
- Validation accuracy, precision, recall, F1-score
- Classification report

### 2. Advanced Model with Hyperparameter Tuning

Train an optimized Random Forest model with GridSearchCV and SMOTE:

```bash
python modelling_tuning.py
```

**What it does:**
- Applies SMOTE to handle class imbalance
- Performs GridSearchCV for hyperparameter tuning
- Tests combinations of:
  - n_estimators: [10, 20, 30]
  - max_depth: [10, 20, 30]
  - min_samples_split: [2, 5, 10, 15, 20]
  - min_samples_leaf: [1, 2, 4, 6, 8]
- Generates and logs visualizations:
  - Confusion matrix
  - ROC curve
  - Feature importance plot
  - Classification report
- Experiment name: `Diabetes_Classification_Skilled` (local) or `Diabetes_Classification_Advanced_DagsHub` (remote)

**Output metrics:**
- All basic metrics
- ROC-AUC score
- Confusion matrix components (TP, TN, FP, FN)
- Best hyperparameters

### 3. Multi-Model Comparison

Compare three different algorithms:

```bash
python modelling_comparison.py
```

**Models compared:**
1. **Random Forest**
   - GridSearchCV parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
   
2. **Gradient Boosting**
   - GridSearchCV parameters: n_estimators, learning_rate, max_depth, min_samples_split
   
3. **Logistic Regression**
   - GridSearchCV parameters: C, penalty, solver

**What it does:**
- Trains all three models with hyperparameter tuning
- Applies SMOTE to training data
- Generates separate visualizations for each model
- Creates comparison chart
- Identifies best performing model
- Experiment name: `Diabetes_Model_Comparison`

**Output:**
- Individual model metrics and artifacts
- Comparison summary with ROC-AUC scores
- Best model identification

### 4. View Experiments in MLflow UI

```bash
mlflow ui
```

Access the MLflow UI at `http://localhost:5000`

**Features:**
- View all experiments and runs
- Compare metrics across runs
- Download models and artifacts
- Visualize metric trends
- Search and filter runs

## Experiment Tracking

### Logged Parameters

- Model type (algorithm name)
- SMOTE applied (boolean)
- Random state
- Hyperparameters from GridSearchCV
- Training/validation sample counts
- Number of features

### Logged Metrics

- Training accuracy
- Validation accuracy
- Validation precision
- Validation recall
- Validation F1-score
- Validation ROC-AUC score
- Confusion matrix components (TP, TN, FP, FN)

### Logged Artifacts

- Trained model (sklearn format)
- Confusion matrix plot (PNG)
- ROC curve plot (PNG)
- Feature importance plot (PNG)
- Classification report (TXT)
- Feature importances (CSV)
- Model comparison chart (PNG, comparison script only)

## DagsHub Integration

The project is integrated with DagsHub for remote experiment tracking and collaboration.

**Repository URL:**
https://dagshub.com/lukaskrisnaaa/mlsystem-submission

**MLflow Tracking UI:**
https://dagshub.com/lukaskrisnaaa/mlsystem-submission.mlflow/

**Experiments Page:**
https://dagshub.com/lukaskrisnaaa/mlsystem-submission/experiments

**Benefits:**
- Cloud-based experiment tracking
- Team collaboration
- Experiment versioning
- Remote model registry
- Data versioning with DVC integration

## Model Performance

### Evaluation Metrics

The models are evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (model discrimination ability)

### Feature Importance

Feature importance is calculated and visualized for tree-based models (Random Forest, Gradient Boosting) and coefficient magnitudes for Logistic Regression.

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

**Features Description:**

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg / height in m^2)
7. **Diabetes Pedigree Function**: Diabetes pedigree function (genetic influence)
8. **Age**: Age in years

**Target Variable:**
- **Outcome**: 0 (No diabetes) or 1 (Diabetes)

### Dataset Citation

If you use this dataset, please cite:

```
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. 
In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). 
IEEE Computer Society Press.
```

## Data Preprocessing

The project uses preprocessed diabetes data located in the `diabetes_preprocessing/` directory:

- **Original dataset**: Downloaded from Kaggle and preprocessed
- **Train-validation split**: Already separated (typically 80-20 or 70-30 split)
- **Scaling**: StandardScaler applied and saved as `scaler.pkl`
- **Missing values**: Handled during preprocessing
- **Features**: All 8 numerical features from the original dataset

## Class Imbalance Handling

The advanced scripts use SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance:

- Generates synthetic samples for the minority class
- Balances the training dataset
- Improves model performance on minority class
- Applied only to training data (not validation)

## Best Practices

### 1. Experiment Naming

Use descriptive experiment and run names:
- Basic models: `Diabetes_Classification_Basic`
- Advanced models: `Diabetes_Classification_Skilled`
- Comparisons: `Diabetes_Model_Comparison`

### 2. Hyperparameter Tuning

Start with a small parameter grid and expand based on results:
- Use cross-validation (cv=3 or cv=5)
- Balance between search space and computation time
- Log all tested parameters

### 3. Model Evaluation

Always evaluate on a separate validation set:
- Never use training data for final evaluation
- Report multiple metrics (not just accuracy)
- Visualize confusion matrix and ROC curve

### 4. Artifact Management

Organize artifacts by model and experiment:
- Use descriptive file names
- Clean up temporary files after logging
- Store important visualizations

## Screenshots

The project includes comprehensive screenshots in the `screenshot/` directory:

- **Local MLflow UI**: Training runs and metrics visualization
- **DagsHub**: Remote experiment tracking interface
- **Base Model**: Baseline model results
- **Tuning Model**: Hyperparameter tuning results

## References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- scikit-learn Documentation: https://scikit-learn.org/stable/
- SMOTE Paper: https://arxiv.org/abs/1106.1813
- DagsHub: https://dagshub.com/docs

