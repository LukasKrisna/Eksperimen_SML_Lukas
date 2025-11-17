from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
import shutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

PREDICTION_SUCCESS = Counter('prediction_success_total', 'Total Successful Predictions')
PREDICTION_FAILURE = Counter('prediction_failure_total', 'Total Failed Predictions')
MEMORY_USAGE_MB = Gauge('system_memory_usage_mb', 'Memory Usage in Megabytes')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of Active Requests')


@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    memory_info = psutil.virtual_memory()
    MEMORY_USAGE_MB.set(memory_info.used / (1024 * 1024))
    
    disk_info = shutil.disk_usage('/')
    DISK_USAGE.set((disk_info.used / disk_info.total) * 100)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    ACTIVE_REQUESTS.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        if response.status_code == 200:
            PREDICTION_SUCCESS.inc()
        else:
            PREDICTION_FAILURE.inc()
        
        ACTIVE_REQUESTS.dec()
        return jsonify(response.json())

    except Exception as e:
        PREDICTION_FAILURE.inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)