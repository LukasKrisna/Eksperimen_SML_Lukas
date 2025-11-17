import requests
import json

def predict(data, url="http://127.0.0.1:8000/predict"):
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    if hasattr(data, "values"):
        formatted_data = data.values.tolist()
    elif data and isinstance(data[0], (list, tuple)):
        formatted_data = [list(row) for row in data]
    else:
        formatted_data = [data]

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": formatted_data
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    sample_patient = [-0.844, -0.876, -1.024, -1.264, -1.259, -1.245, -0.696, -0.956]
    print(predict(sample_patient))
    
if __name__ == "__main__":
    main()

