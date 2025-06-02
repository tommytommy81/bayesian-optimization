import pandas as pd

def load_data(filepath):
    """Load investment decisions data from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the investment decisions data."""
    # Convert columns to appropriate data types
    data['trial_number'] = data['trial_number'].astype(int)
    data['response_time'] = pd.to_timedelta(data['response_time'])
    data['investment_outcome'] = data['investment_outcome'].astype(int)
    
    # Handle missing values if necessary
    data = data.dropna()
    
    return data

def prepare_data(filepath):
    """Load and preprocess the investment data."""
    data = load_data(filepath)
    processed_data = preprocess_data(data)
    return processed_data