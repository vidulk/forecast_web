"""
Data processing utilities for time series data.
"""
import pandas as pd
import re
import numpy as np
from config import DATE_FORMAT_MAP


def detect_granularity(df):
    """
    Attempts to detect the time series granularity (hourly, daily, weekly, monthly)
    from the date column.
    """
    # Find a date column
    date_column = None
    common_date_cols = ['ds', 'date', 'time', 'datetime', 'timestamp', 'day', 'dt']
    for col in common_date_cols:
        if col in df.columns:
            date_column = col
            break
    
    if not date_column:
        return 'daily' # Default if no date column found

    # Try to convert to datetime
    try:
        dates = pd.to_datetime(df[date_column], errors='coerce')
        dates = dates.dropna()
        if len(dates) < 2:
            return 'daily'

        # Calculate the difference and find the most common one
        median_diff = dates.sort_values().diff().median()

        if median_diff <= pd.Timedelta(minutes=90):
            return 'hourly'
        elif median_diff <= pd.Timedelta(hours=36):
            return 'daily'
        elif median_diff <= pd.Timedelta(days=10):
            return 'weekly'
        else:
            return 'monthly'
            
    except Exception:
        return 'daily' # Default on error


def detect_columns(df):
    """Detects date, value, and potential covariate columns."""
    columns = df.columns.tolist()
    
    # Detect date column
    date_column = None
    common_date_cols = ['ds', 'date', 'time', 'datetime', 'timestamp', 'day', 'dt']
    for col in common_date_cols:
        if col in columns:
            date_column = col
            break
    if not date_column and len(columns) > 0:
        date_column = columns[0] # Default to first column

    # Detect value column
    value_column = None
    common_value_cols = ['y', 'value', 'sales', 'revenue', 'count', 'price', 'metric']
    for col in common_value_cols:
        if col in columns and col != date_column and pd.api.types.is_numeric_dtype(df[col]):
            value_column = col
            break
    if not value_column:
        # Fallback: find first numeric column that isn't the date column
        for col in columns:
            if col != date_column and pd.api.types.is_numeric_dtype(df[col]):
                value_column = col
                break
    if not value_column and len(columns) > 1:
        value_column = columns[1] # Default to second column

    # Suggest covariates (all columns that are not date or value)
    guessed_covariates = [
        col for col in columns if col not in [date_column, value_column]
    ]
    
    return date_column, value_column, guessed_covariates


def detect_date_format(df, date_column='ds'):
    """
    Attempts to detect the date format from the first few rows of data.
    Returns a user-friendly format like DD/MM/YYYY.
    """
    # Find the date column if not 'ds'
    if date_column not in df.columns:
        common_date_cols = ['date', 'time', 'datetime', 'timestamp', 'day', 'dt']
        for col in common_date_cols:
            if col in df.columns:
                date_column = col
                break
        else:
            # If no common date column is found, return a default format.
            return "DD/MM/YYYY"

    # Get a sample date value (first non-null)
    sample_dates = df[date_column].dropna()
    sample = str(sample_dates.iloc[0]).strip()
    print(f"[DEBUG] Detecting format for sample date: '{sample}'")

    # Define formats to check, from most specific to most general.
    # Using word boundaries (\b) and end-of-string ($) for precision.
    formats_to_check = {
        r'^\d{4}-\d{2}-\d{2}$': "YYYY-MM-DD",
        r'^\d{2}-\d{2}-\d{4}$': "DD-MM-YYYY",
        r'^\d{2}/\d{2}/\d{4}$': "DD/MM/YYYY",
        r'^\d{4}/\d{2}/\d{2}$': "YYYY/MM/DD",
        r'^\d{1,2}/\d{1,2}/\d{2}$': "DD/MM/YY",        # Updated
        r'^\d{1,2}-\d{1,2}-\d{2}$': "DD-MM-YY",        # Updated
        r'^\d{4}-\d{2}$': "YYYY-MM",
        r'^\d{2}-\w{3}-\d{4}\b': "DD-MON-YYYY",
        r'^\w{3} \d{2}, \d{4}\b': "MON DD, YYYY",
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$': None,  # Ambiguous, needs special handling
    }

    for pattern, fmt in formats_to_check.items():
        if re.match(pattern, sample):
            print(f"[DEBUG] Matched pattern: {pattern}")
            if fmt:  # If format is non-ambiguous, return it
                return fmt
            
            # Handle ambiguous formats like DD/MM/YYYY vs MM/DD/YYYY
            # and DD/MM/YY vs MM/DD/YY
            separator = re.search(r'([/-])', sample).group(1)
            parts = sample.split(separator)
            
            try:
                # Ensure there are three parts to the date
                if len(parts) != 3:
                    continue

                part1 = int(parts[0])
                part2 = int(parts[1])
                year_part = parts[2]
                
                # Explicitly check year length for proper format
                year_format = 'YYYY' if len(year_part) == 4 else 'YY'
                print(f"[DEBUG] Date parts: {part1}{separator}{part2}{separator}{year_part} (year format: {year_format})")

                # Check if the first part is likely a day (>12)
                if part1 > 12:
                    return f"DD{separator}MM{separator}{year_format}"
                # Check if the second part is likely a day (>12)
                elif part2 > 12:
                    return f"MM{separator}DD{separator}{year_format}"
                # If both are <= 12, it's ambiguous. Default to DD/MM as requested.
                else:
                    print("[DEBUG] Ambiguous date (DD/MM vs MM/DD), defaulting to DD/MM.")
                    return f"DD{separator}MM{separator}{year_format}"
            except (ValueError, IndexError) as e:
                print(f"[DEBUG] Error parsing ambiguous date parts: {e}")
                continue # Move to next pattern if parsing fails

    print("[DEBUG] No specific format pattern matched, using default DD/MM/YYYY")
    return "DD/MM/YYYY"  # Default format if no match found


def convert_date_format(user_format):
    """Converts a user-friendly date format to a Python strptime format."""
    return DATE_FORMAT_MAP.get(user_format, "%d/%m/%Y")  # Fallback to DD/MM/YYYY


def prepare_data(data):
    """
    Prepares the data for time series modeling using statsforecast format.
    Ensures 'ds' is a datetime column and converts to 'ds', 'y' to 'y', and adds 'unique_id'.
    Args:
        data (pd.DataFrame): Input DataFrame with 'ds' and 'y' columns.
    Returns:
        pd.DataFrame: DataFrame with columns ['unique_id', 'ds', 'y']
    """
    data = data.rename(columns={'time': 'ds', 'date': 'ds', 'datetime': 'ds', 'dt':'ds'})
    data = data.rename(columns={'sales': 'y', 'rev': 'y', 'num_orders': 'y', 'orders': 'y', 'units_sold': 'y', 'units': 'y', 'revenue': 'y'})
    if 'ds' not in data.columns or 'y' not in data.columns:
        raise ValueError("DataFrame must contain 'ds' and 'y' columns.")
    if len(data) < 7:
        raise ValueError("DataFrame must contain at least seven rows of data.")
    if len(data) > 1000000:
        raise ValueError("DataFrame must contain less than 1,000,000 rows of data.")
        
    data = data.sort_values('ds')
    
    # Preserve covariate columns if they exist
    covariate_cols = [col for col in data.columns if col.startswith('x_')]
    core_cols = ['ds', 'y']
    
    data = data[core_cols + covariate_cols].copy()
    data['unique_id'] = 0
    
    # Reorder columns to have unique_id, ds, y first
    final_cols = ['unique_id', 'ds', 'y'] + covariate_cols
    data = data[final_cols]
    
    return data


def get_baseline_forecast(data, forecast_horizon, granularity):
    """
    Generate a baseline forecast using the mean of the historical data.
    
    Args:
        data (DataFrame): The input time series data.
        forecast_horizon (int): The number of steps to forecast.
        granularity (str): The granularity of the data (e.g., 'D' for daily).
    
    Returns:
        DataFrame: The baseline forecast.
    """
    # Ensure the data is sorted by date
    data = data.sort_values('ds')

    # Resample the data to the desired granularity, using the mean for aggregation
    if granularity == 'D':
        resampled_data = data.resample('D', on='ds').mean().reset_index()
    elif granularity == 'W':
        resampled_data = data.resample('W', on='ds').mean().reset_index()
    elif granularity == 'M':
        resampled_data = data.resample('M', on='ds').mean().reset_index()
    else:
        raise ValueError("Unsupported granularity: must be 'D', 'W', or 'M'")

    # Generate the baseline forecast by repeating the last available value
    baseline_forecast = resampled_data[-forecast_horizon:]['y'].reset_index(drop=True)

    # Create a DataFrame for the forecast
    forecast_dates = pd.date_range(start=resampled_data['ds'].max() + pd.Timedelta(days=1), 
                                    periods=forecast_horizon, 
                                    freq=granularity)
    baseline_forecast_df = pd.DataFrame({'ds': forecast_dates, 'y': baseline_forecast})

    # Calculate accuracy as the mean absolute percentage error (MAPE) on the historical data
    if len(resampled_data) > forecast_horizon:
        historical_values = resampled_data['y'][-(forecast_horizon+1):-1]
        mape = np.mean(np.abs((historical_values - baseline_forecast) / historical_values)) * 100
        accuracy = 100 - mape
    else:
        accuracy = None

    return baseline_forecast_df, accuracy
