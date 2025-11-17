# ML Model Monitoring and Logging System

## Overview

This project implements a comprehensive monitoring and logging system for machine learning models in production. It provides real-time observability through Prometheus metrics collection and Grafana visualization, with automated alerting for system health and model performance issues.

### Key Features

- Real-time metrics collection with Prometheus
- Interactive dashboards with Grafana
- Automated alerting for critical issues
- System resource monitoring (CPU, RAM, Disk usage)
- Model performance tracking (latency, throughput, prediction success/failure)
- RESTful API for model inference
- Flask-based metrics exporter

### Architecture

The system consists of three main components:

1. **Prometheus Exporter** (`prometheus_exporter.py`): Flask application that serves as a proxy between clients and the ML model, collecting metrics
2. **Prometheus Server**: Scrapes and stores metrics data
3. **Grafana**: Visualizes metrics and manages alerts

### Technology Stack

- **Metrics Collection**: Prometheus, prometheus_client
- **Visualization**: Grafana
- **API Framework**: Flask
- **System Monitoring**: psutil
- **Python Version**: 3.12.7

## Project Structure

```
Monitoring_logging/
├── Grafana monitoring/
│   └── [12 monitoring dashboard screenshots]
├── Grafana alerting/
│   ├── high-cpu/
│   │   └── [3 CPU alert screenshots]
│   ├── high-latency/
│   │   └── [3 latency alert screenshots]
│   └── prediction-failure/
│       └── [3 prediction failure alert screenshots]
├── Prometheus monitoring/
│   └── [1 Prometheus dashboard screenshot]
├── inference.py              # Client script for making predictions
├── prometheus_exporter.py    # Flask app with Prometheus metrics
├── prometheus.yml            # Prometheus configuration
├── grafana.ini               # Grafana SMTP configuration for alerting
├── requirements.txt          # Python dependencies
└── serving.png               # Model serving architecture diagram
```

## Requirements

### System Requirements

- Python 3.12.7
- Prometheus (latest version)
- Grafana (latest version)
- Running MLflow model server (on port 5005)

### Python Dependencies

Create a `requirements.txt` file with the following:

```
flask==3.0.0
prometheus-client==0.19.0
requests==2.31.0
psutil==5.9.0
```

## Installation

### 1. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install Prometheus

**macOS:**
```bash
brew install prometheus
```

**Linux:**
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*
```

**Windows:**
Download from https://prometheus.io/download/

### 3. Install Grafana

**macOS:**
```bash
brew install grafana
```

**Linux:**
```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana
```

**Windows:**
Download from https://grafana.com/grafana/download

## Configuration

### 1. Prometheus Configuration

The `prometheus.yml` file is already configured to scrape metrics from the Flask exporter:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "ml_model_exporter"
    static_configs:
    - targets: ["127.0.0.1:8000"]
```

Copy this file to your Prometheus directory or specify it when starting Prometheus.

### 2. Grafana Data Source

After starting Grafana:

1. Navigate to Configuration > Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL to `http://localhost:9090`
5. Click "Save & Test"

### 3. Grafana SMTP Configuration for Alerting

To enable email notifications for alerts, configure SMTP settings in the `grafana.ini` file.

A sample `grafana.ini` configuration file is provided in this project with pre-configured SMTP settings. You can copy the relevant sections to your Grafana configuration.

**Configuration File Location:**

- **macOS (Homebrew)**: `/usr/local/etc/grafana/grafana.ini` or `/opt/homebrew/etc/grafana/grafana.ini`
- **Linux**: `/etc/grafana/grafana.ini`
- **Windows**: `<GRAFANA_INSTALL_DIR>/conf/grafana.ini`

**Basic SMTP Configuration:**

Edit the `[smtp]` section in your Grafana's `grafana.ini`:

```ini
[smtp]
enabled = true
host = smtp.gmail.com:587
user = your-email@gmail.com
password = your-app-password
skip_verify = false
from_address = your-email@gmail.com
from_name = Grafana ML Monitoring

[emails]
welcome_email_on_sign_up = false
templates_pattern = emails/*.html, emails/*.txt
```

**For Gmail:**
- Enable 2-factor authentication on your Google account
- Generate an App Password: https://myaccount.google.com/apppasswords
- Use the App Password in the configuration (not your regular password)

**For Other SMTP Providers:**

Adjust the `host` and `port` based on your provider:
- **Outlook/Office365**: `smtp.office365.com:587`
- **SendGrid**: `smtp.sendgrid.net:587`
- **AWS SES**: `email-smtp.us-east-1.amazonaws.com:587`

**Applying the Configuration:**

Option 1 - Copy entire file:
```bash
# macOS (Homebrew)
sudo cp grafana.ini /usr/local/etc/grafana/grafana.ini
# or
sudo cp grafana.ini /opt/homebrew/etc/grafana/grafana.ini

# Linux
sudo cp grafana.ini /etc/grafana/grafana.ini
```

Option 2 - Copy only SMTP section:
Open the provided `grafana.ini` file, copy the `[smtp]` and `[emails]` sections, and paste them into your Grafana's configuration file.

After editing `grafana.ini`, restart Grafana for changes to take effect:

```bash
# macOS (Homebrew)
brew services restart grafana

# Linux (systemd)
sudo systemctl restart grafana-server

# Windows
# Restart Grafana service from Services manager
```

**Test SMTP Configuration:**

1. Navigate to Alerting > Contact points
2. Click "New contact point"
3. Select "Email" as type
4. Enter a test email address
5. Click "Test" to send a test email
6. Check your inbox for the test email

## Usage

### Step 1: Start the ML Model Server

First, ensure your MLflow model server is running on port 5005:

```bash
mlflow models serve -m /path/to/model -p 5005 --env-manager=local
```

### Step 2: Start Prometheus Exporter

Run the Flask application that exposes metrics:

```bash
python prometheus_exporter.py
```

The exporter will start on `http://127.0.0.1:8000`

### Step 3: Start Prometheus

```bash
prometheus --config.file=prometheus.yml
```

Access Prometheus UI at `http://localhost:9090`

### Step 4: Start Grafana

**macOS/Linux:**
```bash
brew services start grafana
# or
sudo systemctl start grafana-server
```

**Windows:**
```bash
grafana-server.exe
```

Access Grafana UI at `http://localhost:3000` (default credentials: admin/admin)

### Step 5: Make Predictions

Use the inference client to make predictions:

```bash
python inference.py
```

Or make direct API calls:

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "dataframe_split": {
        "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "data": [[-0.844, -0.876, -1.024, -1.264, -1.259, -1.245, -0.696, -0.956]]
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

## Monitored Metrics

### System Metrics

- **CPU Usage** (`system_cpu_usage`): CPU utilization percentage
- **RAM Usage** (`system_ram_usage`): Memory utilization percentage
- **Memory Usage MB** (`system_memory_usage_mb`): Memory usage in megabytes
- **Disk Usage** (`system_disk_usage`): Disk utilization percentage

### Application Metrics

- **Request Count** (`http_requests_total`): Total number of HTTP requests
- **Request Latency** (`http_request_duration_seconds`): HTTP request duration histogram
- **Throughput** (`http_requests_throughput`): Requests per second
- **Active Requests** (`active_requests`): Number of concurrent requests

### Model Performance Metrics

- **Prediction Success** (`prediction_success_total`): Total successful predictions
- **Prediction Failure** (`prediction_failure_total`): Total failed predictions

## Grafana Dashboards

### Main Dashboard

The main dashboard displays:

- Real-time CPU and RAM usage graphs
- Request latency over time
- Prediction success rate
- Active requests gauge
- Disk usage metrics
- Throughput statistics

### Creating a Dashboard

1. In Grafana, click "+" > "Dashboard"
2. Click "Add new panel"
3. Select your Prometheus data source
4. Enter PromQL queries, for example:
   - CPU Usage: `system_cpu_usage`
   - Request Rate: `rate(http_requests_total[1m])`
   - Latency p95: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
   - Success Rate: `rate(prediction_success_total[5m]) / (rate(prediction_success_total[5m]) + rate(prediction_failure_total[5m]))`

## Alerting

The system includes three critical alert rules:

### 1. High CPU Usage Alert

Triggers when CPU usage exceeds 80% for more than 1 minute.

**PromQL:**
```promql
system_cpu_usage > 80
```

### 2. High Latency Alert

Triggers when request latency (95th percentile) exceeds 2 seconds.

**PromQL:**
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
```

### 3. Prediction Failure Alert

Triggers when prediction failure rate exceeds 10%.

**PromQL:**
```promql
rate(prediction_failure_total[5m]) / (rate(prediction_success_total[5m]) + rate(prediction_failure_total[5m])) > 0.1
```

### Setting Up Alerts in Grafana

1. Navigate to Alerting > Alert rules
2. Click "New alert rule"
3. Enter the PromQL query
4. Set evaluation interval and duration
5. Configure notification channels (email, Slack, etc.)
6. Define alert message and severity

### Configuring Email Notifications

After setting up SMTP configuration in `grafana.ini`, configure email contact points:

**Step 1: Create Email Contact Point**

1. Navigate to Alerting > Contact points
2. Click "New contact point"
3. Name: "Email Notifications" (or any preferred name)
4. Integration: Select "Email"
5. Addresses: Enter recipient email addresses (separated by semicolons for multiple recipients)
   - Example: `admin@example.com; devops@example.com`
6. Click "Save contact point"

**Step 2: Create Notification Policy**

1. Navigate to Alerting > Notification policies
2. Click "New specific policy" or edit the default policy
3. Select the email contact point created above
4. Configure grouping and timing:
   - Group by: `alertname` (groups similar alerts)
   - Group wait: `30s` (wait before sending first notification)
   - Group interval: `5m` (wait before sending subsequent notifications)
   - Repeat interval: `4h` (resend if alert still firing)

**Step 3: Link Alerts to Contact Point**

1. Go back to your alert rules
2. Edit each alert rule (High CPU, High Latency, Prediction Failure)
3. In the "Configure labels and notifications" section:
   - Add label: `severity=critical` (for critical alerts)
   - Select the contact point or let it use the default policy
4. Save the alert rule

**Example Alert Rule Configuration:**

```yaml
Alert Rule: High CPU Usage
Query: system_cpu_usage > 80
Evaluation: Every 1m for 1m
Labels:
  - severity: critical
  - component: system
Annotations:
  - summary: High CPU usage detected
  - description: CPU usage is {{ $value }}% (threshold: 80%)
```

**Email Template Example:**

The alert email will contain:
- Alert name and status (Firing/Resolved)
- Timestamp
- Current value
- Query expression
- Labels and annotations
- Link to Grafana dashboard

## API Endpoints

### Metrics Endpoint

```
GET http://127.0.0.1:8000/metrics
```

Returns Prometheus-formatted metrics.

### Prediction Endpoint

```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "dataframe_split": {
    "columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "data": [[-0.844, -0.876, -1.024, -1.264, -1.259, -1.245, -0.696, -0.956]]
  }
}
```

Returns prediction results and updates metrics.

## Workflow

1. Client sends prediction request to Flask exporter (port 8000)
2. Flask exporter forwards request to MLflow model server (port 5005)
3. Flask exporter records metrics (latency, success/failure, system resources)
4. Prometheus scrapes metrics from Flask exporter every 5 seconds
5. Grafana queries Prometheus and displays real-time dashboards
6. Alerts trigger when thresholds are exceeded

## Screenshots

The project includes comprehensive screenshots demonstrating:

- **Grafana Monitoring**: 12 screenshots showing various dashboard views
- **Grafana Alerting**: 9 screenshots showing alert configurations and triggers
  - High CPU alerts
  - High latency alerts
  - Prediction failure alerts
- **Prometheus Monitoring**: Dashboard view
