from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import os

app = Flask(__name__)

# Check if models exist, if not, train them
def check_and_train_models():
    if not os.path.exists("basic_model.pkl"):
        print("Training basic model...")
        import subprocess
        subprocess.run(["python", "train_model.py"])
    
    if not os.path.exists("detailed_model.pkl"):
        print("Training detailed model...")
        import subprocess
        subprocess.run(["python", "train_mlr_model.py"])

# Load trained models
def load_models():
    try:
        basic_model = joblib.load("basic_model.pkl")
        detailed_model = joblib.load("detailed_model.pkl")
        return basic_model, detailed_model
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        check_and_train_models()
        return joblib.load("basic_model.pkl"), joblib.load("detailed_model.pkl")

# Load models at startup
basic_model, detailed_model = load_models()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/basic", methods=["GET", "POST"])
def basic_prediction():
    predicted_time = None
    error = None
    
    if request.method == "POST":
        try:
            speed = float(request.form["speed"])
            if speed <= 0:
                error = "Speed must be greater than zero"
            else:
                # Reshape data for sklearn
                predicted_time = basic_model.predict(np.array([[speed]]))[0]
                # Handle negative predictions (should not happen with proper training)
                predicted_time = max(0, predicted_time)
        except ValueError:
            error = "Please enter a valid number"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template("basic.html", predicted_time=predicted_time, error=error)

@app.route("/detailed", methods=["GET", "POST"])
def detailed_prediction():
    predicted_time = None
    error = None
    
    if request.method == "POST":
        try:
            # Get and validate inputs
            speed = float(request.form["speed"])
            file_size = float(request.form["file_size"])
            active_downloads = int(request.form["active_downloads"])
            device_type = request.form["device_type"]
            
            # Validate inputs
            if speed <= 0:
                error = "Speed must be greater than zero"
            elif file_size <= 0:
                error = "File size must be greater than zero"
            elif active_downloads <= 0:
                error = "Active downloads must be greater than zero"
            else:
                # Check if file size is in MB or GB and convert if necessary
                unit = request.form.get("file_size_unit", "MB")
                if unit == "GB":
                    file_size *= 1024  # Convert GB to MB
                
                # Convert categorical device_type into numerical values
                device_mapping = {"Smartphone": 0, "Tablet": 1, "PC": 2, "Laptop": 3}
                device_encoded = device_mapping.get(device_type, 0)
                
                # Prepare input array for MLR model
                input_data = np.array([[speed, file_size, active_downloads, device_encoded]])
                predicted_time = detailed_model.predict(input_data)[0]
                
                # Convert to seconds if needed and ensure non-negative
                predicted_time = max(0, predicted_time)
        except ValueError:
            error = "Please enter valid numbers"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template("detailed.html", predicted_time=predicted_time, error=error)

if __name__ == "__main__":
    app.run(debug=True)