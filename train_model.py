import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
file_path = "dataset/internet_speed_vs_download_time_1gb.xlsx"
df = pd.read_excel(file_path)

# Step 2: Define Features (X) and Target Variable (y)
X = df[["Internet Speed (Mbps)"]]  # Independent Variable
y = df["Download Time (1GB) (Seconds)"]  # Dependent Variable

# Step 3: Split Data into Training and Testing Sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Save the Model
joblib.dump(model, "models/linear_regression_model.pkl")
print("Model saved successfully!")
