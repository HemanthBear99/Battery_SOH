import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_discharge_curve(df, cycle_id=None, save_dir="plots"):
    """
    Plots the voltage vs capacity discharge curve.
    """
    check_and_create_dir(save_dir)
    plt.figure(figsize=(8,6))
    
    if cycle_id is not None:
        plot_df = df[df['cycle_id'] == cycle_id]
        title = f"Discharge Curve - Cycle {cycle_id}"
        fname = f"discharge_curve_cycle_{cycle_id}.png"
    else:
        plot_df = df
        title = "Discharge Curve (Voltage vs Capacity)"
        fname = "discharge_curve_all.png"

    plt.plot(plot_df['Capacity_Ah'], plot_df['Voltage_measured'], linewidth=2)
    plt.xlabel("Capacity (Ah)")
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

def plot_voltage_distribution(df, save_dir="plots"):
    check_and_create_dir(save_dir)
    plt.figure(figsize=(8,5))
    plt.hist(df['Voltage_measured'], bins=50, color='steelblue', edgecolor='black')
    plt.xlabel("Voltage (V)")
    plt.ylabel("Frequency")
    plt.title("Voltage Distribution (Discharge)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'voltage_distribution.png'))
    plt.close()

def plot_temperature_distribution(df, save_dir="plots"):
    check_and_create_dir(save_dir)
    plt.figure(figsize=(8,5))
    plt.hist(df['Temperature_measured'], bins=40, color='orange', edgecolor='black')
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.title("Temperature Distribution During Discharge")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'temp_distribution.png'))
    plt.close()

def plot_capacity_degradation(agg_df, save_dir="plots"):
    check_and_create_dir(save_dir)
    
    capacities = agg_df['Capacity_Ah_max'].values
    
    plt.figure(figsize=(12,5))
    plt.bar(range(len(capacities)), capacities, color='green')
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity (Ah)")
    plt.title("Capacity Degradation Across Cycles")
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'capacity_degradation.png'))
    plt.close()

def plot_soh_distribution(agg_df, save_dir="plots"):
    check_and_create_dir(save_dir)
    
    soh = agg_df['SOH'].values * 100
    
    plt.figure(figsize=(12,5))
    plt.bar(range(len(soh)), soh, color='teal')
    plt.xlabel("Cycle Number")
    plt.ylabel("SOH (%)")
    plt.title("State of Health (SOH) Distribution")
    plt.ylim(80, 105) # typical battery range
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'soh_distribution.png'))
    plt.close()

def plot_predictions(y_true, y_pred, title="Predictions vs Reality", save_dir="plots", fname="predictions.png"):
    check_and_create_dir(save_dir)
    plt.figure(figsize=(10,6))
    plt.plot(range(len(y_true)), y_true, label='Actual', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()
