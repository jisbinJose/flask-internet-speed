import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create model directory
os.makedirs("model", exist_ok=True)

# Load data
dataset = pd.read_excel('dataset\mlr_dataset.xlsx')

# Extract dependent and independent variables
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Encode categorical values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model

mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"MAE: {mae:.2f}")
print(f"R-squared: {r2:.2f}")