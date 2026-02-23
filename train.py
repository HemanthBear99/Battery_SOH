import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from data_loader import load_nasa_dataset, load_calce_dataset
from feature_engineering import extract_cycle_features, prepare_training_data
from models import build_attention_cnn_bilstm, build_gru_model
from visualizations import plot_capacity_degradation, plot_soh_distribution, plot_predictions

def main():
    parser = argparse.ArgumentParser(description="Battery SOH and RUL Estimator")
    parser.add_argument('--dataset', type=str, choices=['nasa', 'calce'], default='nasa', help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    print(f"--- Starting Pipeline for {args.dataset.upper()} dataset ---")
    
    # 1. Load Data
    if args.dataset == 'nasa':
        data_path = os.path.join("Dataset", "Nasa dataset", "data")
        df = load_nasa_dataset(data_path)
    else:
        data_path = os.path.join("Dataset", "CALCE dataset")
        df = load_calce_dataset(data_path)
        
    if df is None or df.empty:
        print("Data loading failed or returned empty dataframe. Exiting.")
        return

    print("Data loaded successfully. Filtering discharge cycles...")
    # Keep discharge only
    df = df[df['Current_measured'] < 0]
    # Remove tiny current noise
    df = df[df['Current_measured'] < -0.05]

    print("Extracting cycle features...")
    agg_df = extract_cycle_features(df)
    
    if agg_df is None or agg_df.empty:
        print("Feature extraction returned empty dataframe. Exiting.")
        return

    # Visualizations
    print("Generating visual plots of the dataset...")
    plot_capacity_degradation(agg_df, save_dir="plots")
    plot_soh_distribution(agg_df, save_dir="plots")

    print("Preparing training data...")
    X, y_soh, y_rul = prepare_training_data(agg_df)

    # Scale Features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Reshape for 1D CNN / RNN sequence input: (samples, time_steps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Train SOH
    print("\n--- Training SOH Model ---")
    X_tr_soh, X_te_soh, y_tr_soh, y_te_soh = train_test_split(X_scaled, y_soh, test_size=0.2, random_state=42)
    
    model_soh = build_attention_cnn_bilstm((1, X_scaled.shape[2]))
    model_soh.fit(X_tr_soh, y_tr_soh, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=1)
    
    y_pred_soh = model_soh.predict(X_te_soh).flatten()
    print("\n===== SOH RESULTS =====")
    print("R2:", r2_score(y_te_soh, y_pred_soh))
    print("RMSE:", np.sqrt(mean_squared_error(y_te_soh, y_pred_soh)))
    print("MAE:", mean_absolute_error(y_te_soh, y_pred_soh))
    print("MAPE:", mean_absolute_percentage_error(y_te_soh, y_pred_soh))
    
    plot_predictions(y_te_soh, y_pred_soh, title="SOH: Actual vs Predicted", fname="soh_predictions.png")

    # Train RUL 
    print("\n--- Training RUL Model ---")
    X_tr_rul, X_te_rul, y_tr_rul, y_te_rul = train_test_split(X_scaled, y_rul, test_size=0.2, random_state=42)
    
    # We will use the same attention architecture for RUL
    model_rul = build_attention_cnn_bilstm((1, X_scaled.shape[2]))
    model_rul.fit(X_tr_rul, y_tr_rul, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=1)
    
    y_pred_rul = model_rul.predict(X_te_rul).flatten()
    print("\n===== RUL RESULTS =====")
    print("R2:", r2_score(y_te_rul, y_pred_rul))
    print("RMSE:", np.sqrt(mean_squared_error(y_te_rul, y_pred_rul)))
    print("MAE:", mean_absolute_error(y_te_rul, y_pred_rul))
    print("MAPE:", mean_absolute_percentage_error(y_te_rul, y_pred_rul))
    
    plot_predictions(y_te_rul, y_pred_rul, title="RUL: Actual vs Predicted", fname="rul_predictions.png")
    
    print("\nPipeline execution complete! Check the /plots directory for output visuals.")

if __name__ == "__main__":
    main()
