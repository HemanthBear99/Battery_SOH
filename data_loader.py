import os
import pandas as pd

def get_nasa_type1_files(folder_path):
    """
    Scans the NASA dataset folder and returns a list of CSV files 
    that contain raw battery measurements (Type 1).
    """
    type1_files = []
    type1_columns = ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                     'Current_load', 'Voltage_load', 'Time']
    
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} doesn't exist.")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                # Read only first row to check columns
                df = pd.read_csv(file_path, nrows=1)
                cols = df.columns.tolist()
                
                # Detect Type 1 (raw battery data)
                if all(col in cols for col in type1_columns):
                    type1_files.append(file_path)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    return type1_files

def load_nasa_dataset(folder_path):
    """
    Loads all NASA Type 1 files into a single DataFrame.
    """
    files = get_nasa_type1_files(folder_path)
    if not files:
        print("No NASA Type 1 files found.")
        return pd.DataFrame()

    print(f"Found {len(files)} NASA Type 1 files. Loading data...")
    df_list = []
    for file in files:
        try:
            df_temp = pd.read_csv(file)
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def load_calce_dataset(folder_path):
    """
    Loads CALCE dataset CSVs from the provided folder.
    Maps columns to match NASA's format for unified processing.
    """
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} doesn't exist.")
        return pd.DataFrame()
        
    df_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df_temp = pd.read_csv(file_path)
                    
                    # Standardize column names to match NASA format
                    column_mapping = {
                        'V': 'Voltage_measured',
                        'I': 'Current_measured',
                        'T': 'Temperature_measured',
                        'Test_Time_s_': 'Time'
                    }
                    df_temp.rename(columns=column_mapping, inplace=True)
                    
                    # Add remaining dummy columns if needed for shared processing
                    if 'Current_load' not in df_temp.columns:
                        df_temp['Current_load'] = 0.0
                    if 'Voltage_load' not in df_temp.columns:
                        df_temp['Voltage_load'] = 0.0
                        
                    # Filter only columns of interest
                    cols_to_keep = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 
                                    'Current_load', 'Voltage_load', 'Time']
                    
                    if all(col in df_temp.columns for col in cols_to_keep):
                        df_temp = df_temp[cols_to_keep]
                        df_list.append(df_temp)
                except Exception as e:
                    print(f"Error processing CALCE file {file}: {e}")
                    
    if not df_list:
        return pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

if __name__ == "__main__":
    # Test script locally
    nasa_dir = "Dataset/Nasa dataset/data"
    nasa_df = load_nasa_dataset(nasa_dir)
    print(f"NASA Dataset Shape: {nasa_df.shape}")
    
    calce_dir = "Dataset/CALCE dataset"
    calce_df = load_calce_dataset(calce_dir)
    print(f"CALCE Dataset Shape: {calce_df.shape}")
