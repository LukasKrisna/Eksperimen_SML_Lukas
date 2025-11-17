# Machine Learning System: Diabetes Classification

## Overview

This project implements a complete end-to-end machine learning system for diabetes classification using the Pima Indians Diabetes Database. The system encompasses the entire ML lifecycle from data preprocessing to production deployment, including experiment tracking, CI/CD automation, and production monitoring.

**Dataset:** Pima Indians Diabetes Database  
**Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### System Architecture

The project is organized into four main components, each handling a specific phase of the ML lifecycle:

1. **preprocessing**: Data preparation and exploratory analysis
2. **Membangun_model**: Model training and experiment tracking
3. **Workflow-CI**: CI/CD pipeline and automated deployment
4. **Monitoring_logging**: Production monitoring and alerting

### Key Features

- Complete data preprocessing pipeline with outlier handling and normalization
- Multiple ML algorithms comparison (Random Forest, Gradient Boosting, Logistic Regression)
- Hyperparameter tuning with GridSearchCV and SMOTE for class imbalance
- MLflow experiment tracking with DagsHub integration
- Automated CI/CD pipeline with GitHub Actions
- Docker containerization for model serving
- Real-time monitoring with Prometheus and Grafana
- Automated alerting system for system health and model performance

### Technology Stack

- **Language**: Python 3.12.7
- **ML Frameworks**: scikit-learn, imbalanced-learn
- **Experiment Tracking**: MLflow 2.19.0, DagsHub
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Monitoring**: Prometheus, Grafana
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Project Structure

```
root/
├── preprocessing/
│   ├── automate_Lukas.py              # Automated preprocessing script
│   ├── Eksperimen_Lukas.ipynb         # EDA notebook
│   ├── diabetes_preprocessing/        # Preprocessed data output
│   ├── requirements.txt
│   └── README.md
│
├── Membangun_model/
│   ├── modelling.py                   # Basic model training
│   ├── modelling_tuning.py            # Advanced model with tuning
│   ├── modelling_comparison.py        # Multi-model comparison
│   ├── diabetes_preprocessing/        # Preprocessed data
│   ├── mlruns/                        # MLflow tracking directory
│   ├── screenshot/                    # Experiment screenshots
│   ├── requirements.txt
│   └── README.md
│
├── Workflow-CI/
│   ├── .github/workflows/main.yml     # GitHub Actions workflow
│   ├── MLProject/                     # MLflow project
│   │   ├── modelling.py               # Training script
│   │   ├── MLProject                  # MLflow config
│   │   └── conda.yaml                 # Environment config
│   ├── Dockerfile                     # Multi-stage Docker build
│   ├── requirements.txt
│   └── README.md
│
├── Monitoring_logging/
│   ├── prometheus_exporter.py         # Metrics collection
│   ├── inference.py                   # Prediction client
│   ├── prometheus.yml                 # Prometheus config
│   ├── grafana.ini                    # Grafana SMTP config
│   ├── Grafana monitoring/            # Dashboard screenshots
│   ├── Grafana alerting/              # Alert screenshots
│   ├── requirements.txt
│   └── README.md
│
├── diabetes.csv                       # Raw dataset
├── requirements.txt                   # Root dependencies
└── README.md                          # This file
```

## Quick Start

### Prerequisites

- Python 3.12.7 or higher
- Docker Desktop
- Git
- Prometheus (for monitoring)
- Grafana (for monitoring)
- Minimum 4GB RAM
- 3GB free disk space

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/LukasKrisna/Workflow_SML_Lukas.git
cd submission
```

#### 2. Install Root Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Download Dataset

Download the Pima Indians Diabetes Database from Kaggle and place `diabetes.csv` in the root directory:

**Manual Download:**
1. Visit https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place it in the `submission/` directory

**Using Kaggle API:**
```bash
pip install kaggle
kaggle datasets download -d uciml/pima-indians-diabetes-database
unzip pima-indians-diabetes-database.zip
```

## Complete Workflow

### Phase 1: Data Preprocessing

Navigate to the preprocessing directory and run the automated pipeline:

```bash
cd preprocessing
pip install -r requirements.txt
python automate_Lukas.py
```

**What it does:**
- Loads raw diabetes dataset
- Handles outliers using IQR method
- Imputes missing values with median strategy
- Normalizes features using StandardScaler
- Splits data into training (70%) and validation (30%) sets
- Saves preprocessed data to `diabetes_preprocessing/` directory

**Outputs:**
- `X_train.csv`, `X_val.csv`: Feature matrices
- `y_train.csv`, `y_val.csv`: Target vectors
- `scaler.pkl`: Fitted scaler for future use
- `diabetes_preprocessed.csv`: Complete preprocessed dataset

**For detailed information:** See [preprocessing/README.md](preprocessing/README.md)

### Phase 2: Model Training and Experimentation

Navigate to the model building directory and train models:

```bash
cd ../Membangun_model
pip install -r requirements.txt
```

**Option 1: Basic Model**
```bash
python modelling.py
```
Trains a baseline Random Forest without hyperparameter tuning.

**Option 2: Advanced Model with Tuning**
```bash
python modelling_tuning.py
```
Trains Random Forest with GridSearchCV and SMOTE, generates visualizations.

**Option 3: Multi-Model Comparison**
```bash
python modelling_comparison.py
```
Compares Random Forest, Gradient Boosting, and Logistic Regression.

**View Experiments:**
```bash
mlflow ui
```
Access at `http://localhost:5000`

**Optional DagsHub Integration:**
```bash
export MLFLOW_TRACKING_URI='https://dagshub.com/lukaskrisnaaa/mlsystem-submission.mlflow'
export MLFLOW_TRACKING_USERNAME='your-username'
export MLFLOW_TRACKING_PASSWORD='your-token'
```

**For detailed information:** See [Membangun_model/README.md](Membangun_model/README.md)

### Phase 3: CI/CD and Deployment

The CI/CD pipeline automatically trains, builds, and deploys the model:

**GitHub Actions Workflow:**
- Triggered on push to `master` branch
- Trains model using MLflow
- Uploads artifacts to Google Drive
- Builds Docker image
- Pushes to Docker Hub: `lukaskrisna/diabetes-model:latest`

**Manual Docker Deployment:**

```bash
cd ../Workflow-CI

# Pull the pre-built image
docker pull lukaskrisna/diabetes-model:latest

# Run the model server
docker run -p 8080:8080 lukaskrisna/diabetes-model:latest

# Make predictions
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
      "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
      "data": [[-0.844, -0.876, -1.024, -1.264, -1.259, -1.245, -0.696, -0.956]]
    }
  }'
```

**For detailed information:** See [Workflow-CI/README.md](Workflow-CI/README.md)

### Phase 4: Production Monitoring

Set up monitoring and alerting for the deployed model:

```bash
cd ../Monitoring_logging
pip install -r requirements.txt
```

**Step 1: Start ML Model Container**
```bash
docker pull lukaskrisna/diabetes-model:latest
docker run -d -p 5005:8080 --name diabetes-model lukaskrisna/diabetes-model:latest
```

**Step 2: Start Prometheus Exporter**
```bash
python prometheus_exporter.py
```

**Step 3: Start Prometheus**
```bash
prometheus --config.file=prometheus.yml
```
Access at `http://localhost:9090`

**Step 4: Start Grafana**
```bash
brew services start grafana  # macOS
# or
sudo systemctl start grafana-server  # Linux
```
Access at `http://localhost:3000` (default: admin/admin)

**Step 5: Make Predictions**
```bash
python inference.py
```

**Monitored Metrics:**
- System: CPU, RAM, Disk usage
- Application: Request count, latency, throughput
- Model: Prediction success/failure rate

**Alerting:**
- High CPU usage (>80%)
- High latency (>2 seconds)
- Prediction failure rate (>10%)

**For detailed information:** See [Monitoring_logging/README.md](Monitoring_logging/README.md)

## Dataset Information

### Pima Indians Diabetes Database

The dataset contains diagnostic measurements for predicting diabetes in female patients of Pima Indian heritage.

**Characteristics:**
- **Instances**: 768 patients
- **Features**: 8 numerical measurements
- **Target**: Binary classification (0 = No diabetes, 1 = Diabetes)
- **Population**: Females of Pima Indian heritage, at least 21 years old

**Features:**
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration (2-hour oral glucose tolerance test)
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg / height in m^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function (genetic influence)
8. Age: Age in years

**Citation:**
```
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. 
In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). 
IEEE Computer Society Press.
```

## Component Details

### 1. Data Preprocessing (preprocessing/)

Automated data preparation pipeline with:
- Outlier detection and handling (IQR method)
- Missing value imputation (median strategy)
- Feature normalization (StandardScaler)
- Train-validation splitting (70-30)
- Interactive Jupyter notebook for EDA

**Key Files:**
- `automate_Lukas.py`: Production preprocessing script
- `Eksperimen_Lukas.ipynb`: Exploratory analysis notebook

### 2. Model Building (Membangun_model/)

Comprehensive model training with experiment tracking:
- Three training approaches (basic, tuned, comparison)
- MLflow experiment tracking
- DagsHub remote tracking integration
- Automated visualization generation
- SMOTE for class imbalance handling
- GridSearchCV for hyperparameter tuning

**Key Files:**
- `modelling.py`: Baseline model
- `modelling_tuning.py`: Optimized model with SMOTE
- `modelling_comparison.py`: Multi-algorithm comparison

### 3. CI/CD Pipeline (Workflow-CI/)

Automated training and deployment workflow:
- GitHub Actions CI/CD
- MLflow Projects integration
- Docker containerization
- Automated Docker Hub deployment
- Google Drive artifact storage

**Key Files:**
- `.github/workflows/main.yml`: CI/CD workflow
- `Dockerfile`: Multi-stage build configuration
- `MLProject/MLProject`: MLflow project definition

### 4. Production Monitoring (Monitoring_logging/)

Real-time monitoring and alerting system:
- Prometheus metrics collection
- Grafana dashboards and visualization
- Email alerting with SMTP
- System resource monitoring
- Model performance tracking
- Flask-based metrics exporter

**Key Files:**
- `prometheus_exporter.py`: Metrics collection service
- `prometheus.yml`: Prometheus configuration
- `grafana.ini`: Grafana SMTP setup
- `inference.py`: Prediction client

## Dependencies

### Root Dependencies

```
pandas==2.3.3
numpy==2.3.0
scikit-learn==1.7.2
matplotlib==3.10
seaborn==0.13
```

Each component has its own `requirements.txt` with specific dependencies. Install them individually when working on each component.

## Model Performance

The best performing model achieves:
- Validation Accuracy: ~77%
- Precision: ~75%
- Recall: ~72%
- F1-Score: ~73%
- ROC-AUC: ~0.82

Results vary based on the algorithm and hyperparameters used. The model comparison script helps identify the best model for deployment.

## Docker Hub

Pre-built Docker images are available on Docker Hub:

**Repository:** https://hub.docker.com/r/lukaskrisna/diabetes-model

**Pull the latest image:**
```bash
docker pull lukaskrisna/diabetes-model:latest
```

## GitHub Repository

**Workflow Repository:** https://github.com/LukasKrisna/Workflow_SML_Lukas

## DagsHub Integration

**Repository:** https://dagshub.com/lukaskrisnaaa/mlsystem-submission  
**MLflow UI:** https://dagshub.com/lukaskrisnaaa/mlsystem-submission.mlflow/  
**Experiments:** https://dagshub.com/lukaskrisnaaa/mlsystem-submission/experiments

## Development Workflow

### For Local Development

1. **Data Exploration:**
   ```bash
   cd preprocessing
   jupyter notebook Eksperimen_Lukas.ipynb
   ```

2. **Experiment with Models:**
   ```bash
   cd Membangun_model
   python modelling_comparison.py
   mlflow ui
   ```

3. **Test Docker Locally:**
   ```bash
   cd Workflow-CI
   mlflow run MLProject --env-manager=local
   # Get run_id and build Docker
   mlflow models build-docker --model-uri "runs:/<RUN_ID>/model" --name "diabetes-model"
   docker run -p 8080:8080 diabetes-model
   ```

4. **Test Monitoring:**
   ```bash
   cd Monitoring_logging
   python prometheus_exporter.py &
   prometheus --config.file=prometheus.yml &
   python inference.py
   ```

### For Production Deployment

1. Push changes to `master` branch
2. GitHub Actions automatically:
   - Trains the model
   - Builds Docker image
   - Pushes to Docker Hub
3. Deploy the image from Docker Hub
4. Set up monitoring with Prometheus and Grafana

## Project Timeline

1. **Data Preprocessing**: Prepare and clean the dataset
2. **Model Development**: Train and compare multiple algorithms
3. **Experiment Tracking**: Log all experiments with MLflow
4. **CI/CD Setup**: Automate training and deployment
5. **Containerization**: Package model in Docker
6. **Production Monitoring**: Deploy monitoring and alerting

## References

- **Dataset**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **scikit-learn**: https://scikit-learn.org/stable/
- **Docker**: https://docs.docker.com/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **GitHub Actions**: https://docs.github.com/en/actions
- **DagsHub**: https://dagshub.com/docs
