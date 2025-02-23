from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/linear_regression_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_time = None
    
    if request.method == "POST":
        try:
            speed = float(request.form["speed"])
            predicted_time = model.predict(np.array([[speed]]))[0]
        except:
            predicted_time = "Invalid input. Please enter a number."

    return render_template("index.html", predicted_time=predicted_time)

if __name__ == "__main__":
    app.run(debug=True)
