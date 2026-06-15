import pandas as pd
import os

# Import functions from your modular files
from src.data_preprocessing import clean_data
from src.feature_engineering import preprocess_titanic
from src.prepare_data import prepare_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualize import visualize_data
def run_full_pipeline():
    # 1. Ensure required directories exist to avoid FileNotFoundError
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    
    print("--- Starting Titanic ML Pipeline ---")
    
    # 2. Load Raw Data (Make sure you have your raw train.csv in this path)
    # If your file path is different, update it here:
    raw_data_path = "data/raw/train.csv" 
    if not os.path.exists(raw_data_path):
        print(f"Error: Please place your raw dataset at {raw_data_path}")
        return
        
    df = pd.read_csv(raw_data_path)
    
    # 3. Clean Data (data_preprocessing.py)
    print("Cleaning data...")
    cleaned_df = clean_data(df)
    
    # 4. Run Visualizations (visualize.py)
    print("Generating and saving plots...")
    visualize_data(cleaned_df)
    
    # 5. Feature Engineering (feature_engineering.py)
    print("Engineering features...")
    processed_df = preprocess_titanic(cleaned_df)
    
    # 6. Prepare, Scale, and Split Data (prepare_data.py)
    print("Splitting and scaling data...")
    prepare_data(processed_df)
    
    # 7. Train and Evaluate Models (train_model.py & evaluate_model.py)
    print("Evaluating models...")
    metrics = evaluate_model()
    
    print("\n--- Pipeline Execution Complete! ---")
    print("Results:")
    for model_name, score_dict in metrics.items():
        print(f"\nModel: {model_name.upper()}")
        for metric, value in score_dict.items():
            if metric != "conf_matrix":
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    run_full_pipeline()