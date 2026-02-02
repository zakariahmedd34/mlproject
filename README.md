# Student Performance Prediction - End-to-End ML Project

## Overview
This project is an end-to-end Machine Learning pipeline designed to predict student performance (specifically Math Scores) based on various demographic and educational factors. It provides a user-friendly web interface built with Flask to input student data and receive real-time predictions.

## Project Structure
```text
mlproject/
├── app.py              # Flask Application entry point
├── requirements.txt    # Project dependencies
├── setup.py           # Packaging script
├── Dockerfile         # Docker containerization
├── artifacts/         # Trained models and preprocessed data
├── src/               # Source code
│   ├── components/    # Data ingestion, transformation, and trainer
│   ├── pipeline/      # Training and prediction pipelines
│   ├── logger.py      # Custom logging
│   ├── exception.py   # Custom exception handling
│   └── utils.py       # Helper functions
├── templates/         # HTML files for Flask
└── static/            # CSS and JS for Flask
```

## Features
- **Data Ingestion**: Automated scripts to fetch data.
- **Data Transformation**: Robust preprocessing using Scikit-learn pipelines.
- **Model Training**: Evaluation of multiple models (CatBoost, XGBoost, Random Forest, etc.) to find the best performer.
- **Prediction Pipeline**: Clean interface to process input and return results.
- **Web App**: Interactive UI for end-users to input data and see predictions.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd mlproject
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. To start the Flask application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Use the interface to enter student information and get the predicted Math score.

## Docker Deployment
To build and run the project using Docker:
```bash
docker build -t mlproject .
docker run -p 5000:5000 mlproject
```

## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework.
- **Scikit-learn**: Machine learning utilities.
- **Pandas/Numpy**: Data manipulation.
- **XGBoost/CatBoost**: Advanced Gradient Boosting models.
- **Docker**: For containerization and deployment.