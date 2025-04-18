import pandas as pd
from darts import TimeSeries
from darts.models import Theta, NaiveSeasonal

def prepare_data(data):
    """
    Prepares the data for time series modeling using darts TimeSeries format.
    Ensures 'dt' is a datetime column and converts to darts TimeSeries.
    
    Args:
        data (pd.DataFrame): Input DataFrame with 'dt' and 'value' columns.
    Returns:
        TimeSeries: A darts TimeSeries object ready for modeling.
    """

    # rename any column named time, date, or datetime to dt
    data = data.rename(columns={'time': 'dt', 'date': 'dt', 'datetime': 'dt'})
    data = data.rename(columns={'sales': 'value', 'rev': 'value', 'numm_orders': 'value', 'orders': 'value', 'units_sold': 'value', 'units': 'value', 'revenue': 'value'})
    
    if 'dt' not in data.columns or 'value' not in data.columns:
        raise ValueError("DataFrame must contain 'dt' and 'value' columns.")
    
    if len(data) < 7:
        raise ValueError("DataFrame must contain at least seven rows of data.")
    
    if len(data) > 1000000:
        raise ValueError("DataFrame must contain less than 1,000,000 rows of data.")
    
    # Convert 'dt' to datetime and sort the data
    try:
        data['dt'] = pd.to_datetime(data['dt'], format='%d/%m/%y')
    except:
        pass

    # reformat dt column to be in YYYY-MM-DD format
    data['dt'] = data['dt'].dt.strftime('%Y-%m-%d')

    data = data.sort_values('dt')
    # Convert to darts TimeSeries format
    series = TimeSeries.from_dataframe(
        df=data,
        time_col='dt',
        value_cols='value',
        freq=None  # Auto-infer frequency
    )
    
    return series

def forecast_with_theta(data, forecast_steps=10):
    """
    Creates forecasts using the Theta model from darts.
    
    Args:
        data (TimeSeries): Input time series in darts TimeSeries format
        forecast_steps (int): Number of steps to forecast
    
    Returns:
        pd.DataFrame: Forecast results with datetime index
    """
    # Initialize and fit the model
    model = Theta()  
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def calculate_cv_accuracy(series, forecast_function, forecast_horizon=10, num_folds=3, stride=1):
    """
    Calculates cross-validated accuracy metrics (MAE and MAPE) for a given forecasting function.
    
    Args:
        data (pd.DataFrame): Input DataFrame with 'dt' and 'value' columns
        forecast_function (callable): Function that takes (data, forecast_steps) and returns forecast DataFrame
        forecast_horizon (int): Number of steps to forecast in each fold
        num_folds (int): Number of cross-validation folds to use
        stride (int): Step size between folds
    
    Returns:
        dict: Dictionary containing 'mae' and 'mape' accuracy metrics
    """
    from darts.metrics import mae, mape
        
    # Ensure we have enough data for the specified folds and horizon
    min_required_length = num_folds * stride + forecast_horizon
    if len(series) < min_required_length:
        raise ValueError(f"Time series too short for {num_folds} folds with horizon {forecast_horizon}")
    
    mae_scores = []
    mape_scores = []
    
    # Perform rolling-window cross-validation
    max_train_size = len(series) - forecast_horizon
    
    for i in range(num_folds):
        # Calculate cutoff point for this fold
        cutoff = max_train_size - (i * stride)
        if cutoff <= forecast_horizon:
            break  # Stop if we don't have enough data for this fold
        
        # Split data into train and test
        train = series[:cutoff]
        test = series[cutoff:cutoff + forecast_horizon]
        
        # Generate forecast using the provided forecast function
        forecast_df = forecast_function(train, forecast_horizon)
        
        # Convert forecast back to TimeSeries for evaluation
        forecast_ts = TimeSeries.from_dataframe(
            forecast_df, 
            time_col='dt', 
            value_cols='value',
            freq=None
        )
        
        # Calculate metrics
        try:
            mae_val = mae(test, forecast_ts)
            mape_val = mape(test, forecast_ts)
            
            mae_scores.append(mae_val)
            mape_scores.append(mape_val)
        except Exception as e:
            print(f"Error calculating metrics for fold {i+1}: {str(e)}")
            continue
    
    # Calculate average metrics
    avg_mae = sum(mae_scores) / len(mae_scores) if mae_scores else None
    avg_mape = sum(mape_scores) / len(mape_scores) if mape_scores else None
    
    return {
        'mae': avg_mae,
        'mape': avg_mape,
        'individual_mae': mae_scores,
        'individual_mape': mape_scores
    }

# Example usage:
# cv_metrics = calculate_cv_accuracy(data, forecast_with_theta, forecast_horizon=10, num_folds=3)
def baseline_forecast(data, forecast_steps=10):
    
    # Initialize and fit the model
    model = NaiveSeasonal(K=7)
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def get_accuracy(forecast_df):
    return 0