import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path='diabetes.csv'):
    """
    Load diabetes dataset from CSV file.
    
    Args:
        file_path (str): Path to the diabetes CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def handle_outliers(df, columns=['Insulin', 'DiabetesPedigreeFunction']):
    """
    Handle outliers using IQR method for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to handle outliers for
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    df_processed = df.copy()
    
    for column_name in columns:
        if column_name in df_processed.columns:
            Q1 = np.percentile(df_processed[column_name], 25, interpolation='midpoint')
            Q3 = np.percentile(df_processed[column_name], 75, interpolation='midpoint')
            
            IQR = Q3 - Q1
            low_lim = Q1 - 1.5 * IQR
            up_lim = Q3 + 1.5 * IQR
            
            df_processed[column_name] = np.where(df_processed[column_name] < low_lim, low_lim, df_processed[column_name])
            df_processed[column_name] = np.where(df_processed[column_name] > up_lim, up_lim, df_processed[column_name])
            
            print(f"Outliers handled for column: {column_name}")
    
    return df_processed


def handle_missing_values(df, cols_invalid_zero=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]):
    """
    Handle missing values by replacing zeros with NaN and then filling with median.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cols_invalid_zero (list): Columns where 0 values are invalid and should be treated as missing
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    df_processed = df.copy()
    
    df_processed[cols_invalid_zero] = df_processed[cols_invalid_zero].replace(0, np.nan)
    
    df_processed[cols_invalid_zero] = df_processed[cols_invalid_zero].fillna(df_processed[cols_invalid_zero].median())
    
    print("Missing values handled successfully")
    return df_processed


def normalize_features(df, num_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                    'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']):
    """
    Normalize numerical features using Standard Scaler.
    
    Args:
        df (pd.DataFrame): Input dataframe
        num_cols (list): List of numerical columns to normalize
        
    Returns:
        tuple: (normalized_dataframe, fitted_scaler)
    """
    df_processed = df.copy()
    
    scaler = StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    
    print("Feature normalization completed")
    return df_processed, scaler


def prepare_features_target(df, target_column='Outcome'):
    """
    Separate features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (features_dataframe, target_series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def preprocess_diabetes_data(file_path='diabetes.csv', test_size=0.3, random_state=42, save_scaler=True):
    """
    Complete preprocessing pipeline for diabetes dataset.
    
    Args:
        file_path (str): Path to the diabetes CSV file
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility
        save_scaler (bool): Whether to save the fitted scaler
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler)
    """
    print("Starting diabetes data preprocessing pipeline...")
    
    df = load_data(file_path)
    
    df = handle_outliers(df)
    
    df = handle_missing_values(df)
    
    df, scaler = normalize_features(df)
    
    X, y = prepare_features_target(df)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Data split completed:")
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'diabetes_preprocessing')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if save_scaler:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to '{scaler_path}'")
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    
    preprocessed_df = pd.concat([X, y], axis=1)
    preprocessed_df.to_csv(os.path.join(output_dir, 'diabetes_preprocessed.csv'), index=False)
    
    print(f"Preprocessed data saved to '{output_dir}/' folder")
    print("Preprocessing pipeline completed successfully!")
    
    return X_train, X_val, y_train, y_val, scaler


def load_preprocessed_data(data_dir=None):
    """
    Load preprocessed data from saved files.
    
    Args:
        data_dir (str): Directory containing preprocessed data files. 
                       If None, uses diabetes_preprocessing in same directory as script.
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler)
    """
    try:
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'diabetes_preprocessing')
        
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
        y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).squeeze()
        
        with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        print("Preprocessed data loaded successfully")
        return X_train, X_val, y_train, y_val, scaler
        
    except FileNotFoundError as e:
        print(f"Error loading preprocessed data: {e}")
        print("Please run the preprocessing pipeline first")
        return None


if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    X_train, X_val, y_train, y_val, scaler = preprocess_diabetes_data(
        file_path='../diabetes.csv',
        test_size=0.3,
        random_state=42
    )
