# Titanic Survival Prediction Pipeline

A clean, modular end-to-end machine learning pipeline built with Python and scikit-learn to predict passenger survival on the Titanic dataset. The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and visualization.

---

## 📌 Project Overview

The objective of this project is to predict whether a passenger survived the Titanic disaster using demographic and travel-related features.

Rather than building everything inside a single notebook, the project is organized into reusable Python modules that separate each stage of the machine learning workflow.

---

## ⭐ Project Highlights

* End-to-end machine learning workflow
* Modular and reusable code structure
* Automated data preprocessing and feature engineering
* Multiple machine learning models for comparison
* Exploratory data analysis and visualization
* Reproducible pipeline design
* Clean software engineering practices

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

---

## 📊 Dataset

This project uses the Titanic dataset, which contains passenger demographic and travel information such as:

* Age
* Sex
* Passenger Class (Pclass)
* Fare
* Family Relationships
* Embarkation Port

### Target Variable

* Survived (0 = No, 1 = Yes)

---

## 📂 Project Directory Structure

```text
├── data/
│   └── processed/
│       ├── clean_train.csv
│       └── clean_test.csv
│
├── outputs/
│   └── figures/
│
├── data_preprocessing.py
├── feature_engineering.py
├── prepare_data.py
├── train_model.py
├── evaluate_model.py
├── visualize.py
├── run_pipeline.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Pipeline Architecture

### 1. Data Preprocessing (`data_preprocessing.py`)

* Handles missing values
* Fills missing Age values using the median
* Fills missing Embarked values using the mode
* Drops irrelevant columns such as Cabin
* Removes duplicate records

### 2. Feature Engineering (`feature_engineering.py`)

* Creates Family_size feature
* Creates IsAlone feature
* Encodes categorical variables
* Applies one-hot encoding where appropriate
* Removes non-predictive columns such as Name, Ticket, and PassengerId

### 3. Data Preparation (`prepare_data.py`)

* Selects model features
* Splits data into training and testing sets
* Applies feature scaling using StandardScaler
* Prevents data leakage by fitting the scaler only on training data

### 4. Model Training (`train_model.py`)

The following machine learning models are trained:

* Logistic Regression
* Decision Tree Classifier

### 5. Model Evaluation (`evaluate_model.py`)

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 6. Data Visualization (`visualize.py`)

Generates exploratory visualizations including:

* Survival by Gender
* Survival by Passenger Class
* Age Distribution
* Feature Correlation Heatmap

All generated figures are automatically saved to:

```text
outputs/figures/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the complete machine learning pipeline with:

```bash
python run_pipeline.py
```

Or execute each module individually:

```bash
python data_preprocessing.py
python feature_engineering.py
python prepare_data.py
python train_model.py
python evaluate_model.py
python visualize.py
```

---

## 📈 Results

The trained models are evaluated on unseen test data using standard classification metrics.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The project is designed to compare multiple classification models and identify the best-performing approach for Titanic survival prediction.

---

## 🧠 Skills Demonstrated

* Data Cleaning and Preprocessing
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Data Visualization
* Machine Learning Model Development
* Classification Algorithms
* Model Evaluation
* Python Programming
* Scikit-learn
* Modular Pipeline Design
* Reproducible Machine Learning Workflows

---

## 🔮 Future Improvements

* Hyperparameter Tuning with GridSearchCV
* Random Forest Classifier
* XGBoost Integration
* Cross-Validation
* Automated Scikit-learn Pipelines
* Interactive Web App Deployment with Streamlit

---

## 👨‍💻 Author

**Habtamu Yibeltal**

Aspiring Machine Learning Engineer focused on building practical machine learning systems, data-driven solutions, and end-to-end AI projects.
