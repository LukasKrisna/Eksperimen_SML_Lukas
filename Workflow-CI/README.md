# Workflow-CI: Diabetes Classification ML Pipeline

## Overview

This project implements a complete CI/CD pipeline for a diabetes classification machine learning model using MLflow, GitHub Actions, and Docker. The pipeline automatically trains a Random Forest classifier, tracks experiments with MLflow, builds a Docker image, and deploys it to Docker Hub.

### Key Features

- Automated model training using MLflow projects
- Random Forest classifier for diabetes prediction
- Continuous Integration/Continuous Deployment with GitHub Actions
- Docker containerization for model serving
- Automated deployment to Docker Hub
- Google Drive integration for artifact storage
- Multi-stage Docker build for optimized image size

### Technology Stack

- **Machine Learning**: scikit-learn, pandas, numpy
- **Experiment Tracking**: MLflow 2.19.0
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Python Version**: 3.12.7

## Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── main.yml          # GitHub Actions CI/CD workflow
├── MLProject/
│   ├── diabetes_preprocessing/
│   │   ├── diabetes_preprocessed.csv
│   │   ├── scaler.pkl
│   │   ├── X_train.csv
│   │   ├── X_val.csv
│   │   ├── y_train.csv
│   │   └── y_val.csv
│   ├── MLProject             # MLflow project configuration
│   ├── conda.yaml            # Conda environment specification
│   ├── modelling.py          # Model training script
│   └── upload_to_gdrive.py   # Google Drive upload script
├── Dockerfile                # Multi-stage Docker build
├── .gitignore
└── link.txt                  # Links to Docker Hub and GitHub repo
```

## Requirements

### Local Development

Create a `requirements.txt` file with the following dependencies:

```
mlflow==2.19.0
scikit-learn==1.6.0
pandas==2.2.3
numpy==2.2.1
joblib
cloudpickle==3.1.0
dagshub==0.5.1
psutil==5.9.0
scipy==1.14.1
```

### System Requirements

- Python 3.12.7
- Docker (for containerization)
- Git

### GitHub Secrets (for CI/CD)

The following secrets must be configured in your GitHub repository:

- `DOCKER_HUB_USERNAME`: Docker Hub username
- `DOCKER_HUB_ACCESS_TOKEN`: Docker Hub access token
- `GDRIVE_CREDENTIALS`: Google Drive API credentials (JSON)
- `GDRIVE_FOLDER_ID`: Target Google Drive folder ID

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LukasKrisna/Workflow_SML_Lukas.git
cd Workflow-CI
```

### 2. Set Up Python Environment

Using pip:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f MLProject/conda.yaml
conda activate mlflow-env
```

### 3. Verify Installation

```bash
python -c "import mlflow; import sklearn; print('Installation successful')"
```

## Usage

### Running Locally

#### Train the Model

```bash
cd MLProject
mlflow run . --env-manager=local
```

This will:
- Load preprocessed diabetes data
- Train a Random Forest classifier
- Log metrics and model to MLflow
- Display training and validation metrics

#### View MLflow UI

```bash
mlflow ui
```

Access the MLflow UI at `http://localhost:5000` to view experiments and metrics.

### Using Docker

#### Pull the Pre-built Image

```bash
docker pull lukaskrisna/diabetes-model:latest
```

#### Run the Model Server

```bash
docker run -p 8080:8080 -v /path/to/model:/opt/ml/model lukaskrisna/diabetes-model:latest
```

The model will be available at `http://localhost:8080`.

#### Make Predictions

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
      "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
      "data": [[-0.844, -0.876, -1.024, -1.264, -1.259, -1.245, -0.696, -0.956]]
    }
  }'
```

### Building Docker Image Locally

```bash
# First, train the model and get the run_id
mlflow run MLProject --env-manager=local

# Get the latest run_id from MLflow UI or use:
python -c "import mlflow; client = mlflow.tracking.MlflowClient(); runs = client.search_runs(experiment_ids=['0'], order_by=['start_time DESC'], max_results=1); print(runs[0].info.run_id)"

# Build Docker image
mlflow models build-docker --model-uri "runs:/<RUN_ID>/model" --name "diabetes-model"

# Run the container
docker run -p 8080:8080 diabetes-model
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yml`) automatically:

1. Checks out the code
2. Sets up Python 3.12.7
3. Installs dependencies
4. Runs the MLflow project
5. Retrieves the latest run ID
6. Uploads artifacts to Google Drive
7. Builds a Docker image
8. Pushes the image to Docker Hub

### Triggering the Pipeline

The pipeline runs automatically on:
- Push to `master` branch
- Pull requests to `master` branch

## Model Details

### Algorithm

Random Forest Classifier with default hyperparameters (scikit-learn)

### Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

### Preprocessing

The dataset is preprocessed and split into:
- Training set: `X_train.csv`, `y_train.csv`
- Validation set: `X_val.csv`, `y_val.csv`
- Scaler: `scaler.pkl` (for feature scaling)

## Docker Image

The Docker image uses a multi-stage build approach:

- **Builder stage**: Installs build dependencies and Python packages
- **Production stage**: Lightweight image with only runtime dependencies
- **Optimizations**: Memory limits, environment variables for performance

Docker Hub: `https://hub.docker.com/r/lukaskrisna/diabetes-model`

## Links

- Docker Hub: https://hub.docker.com/r/lukaskrisna/diabetes-model
- GitHub Repository: https://github.com/LukasKrisna/Workflow_SML_Lukas
