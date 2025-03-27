# Download Time Predictor

A Flask web application that predicts download times based on internet speed and file size.

## Features

- **Basic Prediction**: Predict download time based solely on internet speed
- **Detailed Prediction**: Advanced prediction considering internet speed, file size, active downloads, and device type
- **95% Accuracy Rate**: High accuracy prediction model
- **Modern UI**: Clean, responsive user interface

## Technology Stack

- Python 3.12
- Flask
- Scikit-learn
- HTML/CSS
- Deployed on Render

## Setup and Installation

1. Clone the repository:
   ```
   git clone [your-repository-url]
   cd download-time-predictor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Unix/MacOS
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Deployment

This application is deployed on Render. For deployment:

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Configure the build settings with:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

## Models

- **Basic Model**: Simple linear regression model
- **Detailed Model**: Multiple linear regression model with additional features

## License

[MIT License](LICENSE) 