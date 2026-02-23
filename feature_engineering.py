import numpy as np

def extract_cycle_features(df):
    """
    Calculates SOH, RUL, and aggregates cycle-level features.
    Assumes df has columns: 'Time', 'Current_measured', 'Voltage_measured', 'Temperature_measured'
    """
    if df.empty:
        return None

    # Sort by time to ensure differential calculations are correct
    df = df.sort_values(by='Time')
    
    # 1. Compute Cycle ID
    # In continuous logs, time dropping back implies a new cycle
    df['cycle_id'] = (df['Time'].diff() < 0).cumsum()
    
    # 2. Compute Capacity Ah per Cycle
    df['dt'] = df['Time'].diff().fillna(0).clip(lower=0)
    df['discharge_current'] = df['Current_measured'].where(df['Current_measured'] < 0, 0)
    df['Capacity_Ah_contrib'] = (df['discharge_current'].abs() * df['dt']) / 3600
    df['Capacity_Ah'] = df.groupby('cycle_id')['Capacity_Ah_contrib'].cumsum()

    cycle_capacity = df.groupby('cycle_id')['Capacity_Ah'].max()
    if cycle_capacity.empty:
        return None
        
    rated_capacity = cycle_capacity.max()
    if rated_capacity == 0:
        rated_capacity = 1.0 # avoid div by zero
        
    cycle_soh = cycle_capacity / rated_capacity
    df['SOH'] = df['cycle_id'].map(cycle_soh).clip(0, 1)

    # 3. Aggregate Features per Cycle
    agg_df = df.groupby('cycle_id').agg({
        'Voltage_measured': ['min', 'max', 'mean', 'std'],
        'Current_measured': ['min', 'max', 'mean', 'std'],
        'Temperature_measured': ['min', 'max', 'mean', 'std'],
        'Current_load': ['mean'],
        'Voltage_load': ['mean'],
        'Capacity_Ah': ['max']
    })

    # Flatten MultiIndex columns
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df['SOH'] = cycle_soh.values

    # 4. Compute Remaining Useful Life (RUL)
    total_cycles = len(agg_df)
    agg_df['RUL'] = total_cycles - agg_df.index - 1
    
    # Clean up NaNs from std dev aggregations of single-row cycles
    agg_df.fillna(0, inplace=True)
    
    return agg_df

def prepare_training_data(agg_df):
    """
    Splits the aggregated dataframe into Features (X) and Targets (SOH, RUL)
    """
    if agg_df is None or agg_df.empty:
        return None, None, None
        
    ALL_FEATURES = [c for c in agg_df.columns if c not in ['SOH', 'RUL']]
    
    X = agg_df[ALL_FEATURES].values
    y_soh = agg_df['SOH'].values
    y_rul = agg_df['RUL'].values
    
    return X, y_soh, y_rul
