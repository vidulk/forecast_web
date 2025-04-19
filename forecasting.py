import pandas as pd
from darts import TimeSeries
from darts.models import Theta, NaiveSeasonal, NaiveDrift

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

def select_forecasting_model(data, domain=None):
    """
    Selects the best forecasting model based on data characteristics and domain.
    
    Args:
        data (TimeSeries): The prepared time series data
        domain (str, optional): Domain of the data (e.g., 'retail', 'finance', 'energy')
    
    Returns:
        callable: The recommended forecasting function
    """
    # Extract data characteristics
    data_length = len(data)
    frequency = data.freq_str
    has_seasonality = check_seasonality(data)
    has_trend = check_trend(data)
    is_stationary = check_stationarity(data)
    complexity = check_complexity(data)
    missing_ratio = check_missing_values(data)
    
    # Handle datasets with significant missing values
    if missing_ratio > 0.1:  # More than 10% missing values
        print(f"Warning: Dataset contains {missing_ratio:.1%} missing values. Using robust methods.")
        # Prophet and LightGBM are relatively robust to missing values
        if data_length > 100:
            return lambda data, forecast_steps: forecast_with_prophet(data, forecast_steps)
        else:
            return lambda data, forecast_steps: forecast_with_theta(data, forecast_steps)
    
    # Make decisions based on stationarity
    if not is_stationary:
        print("Data is non-stationary.")
        if complexity > 0.6:
            # For complex non-stationary data, use methods that handle differencing
            if data_length > 200:
                return lambda data, forecast_steps: forecast_with_autoarima(data, forecast_steps)
            else:
                # For shorter series, Theta often works well with non-stationary data
                return lambda data, forecast_steps: forecast_with_theta(data, forecast_steps)
    else:
        print("Data is stationary.")
        # For stationary data, simpler methods may work well
        if has_seasonality and data_length > 100:
            return lambda data, forecast_steps: forecast_with_ets(data, forecast_steps)
    
    # Data length-based decisions (existing logic but adjusted)
    if data_length < 30:
        # Very short series - use simple models
        return lambda data, forecast_steps: forecast_with_theta(data, forecast_steps)
    
    elif data_length < 100:
        # Short series - statistical methods generally work better
        if has_seasonality:
            # Use ETS for short seasonal data
            return lambda data, forecast_steps: forecast_with_ets(data, forecast_steps)
        else:
            # Use Theta for short non-seasonal data
            return lambda data, forecast_steps: forecast_with_theta(data, forecast_steps)
    
    elif data_length < 1000:
        # Medium series - statistical or simple ML methods
        if has_seasonality and has_trend:
            # Prophet handles seasonality and trend well in this range
            return lambda data, forecast_steps: forecast_with_prophet(data, forecast_steps)
        elif complexity > 0.7:  # Complex patterns detected
            # AutoARIMA can capture complex patterns
            return lambda data, forecast_steps: forecast_with_autoarima(data, forecast_steps)
        else:
            # Theta is robust for medium-sized data
            return lambda data, forecast_steps: forecast_with_theta(data, forecast_steps)
    
    # Domain-specific overrides
    elif domain == 'web_traffic' or domain == 'clickstream':
        return lambda data, forecast_steps: forecast_with_prophet(data, forecast_steps)
    elif domain == 'finance' and not is_stationary:
        return lambda data, forecast_steps: forecast_with_autoarima(data, forecast_steps)
    else:
        # Long series - ML methods often work better
        # LightGBM works well for retail/demand forecasting with many datapoints
        return lambda data, forecast_steps: forecast_with_lightgbm(data, forecast_steps)
    
def check_seasonality(data, max_lag=None):
    """
    Detects if the time series has significant seasonality.
    
    Args:
        data (TimeSeries): Input time series
        max_lag (int, optional): Maximum lag to check for seasonality
        
    Returns:
        bool: True if seasonality is detected
    """
    from statsmodels.tsa.stattools import acf
    import numpy as np
    
    values = data.values().flatten()
    
    if max_lag is None:
        # Try to infer seasonality period from frequency
        freq = data.freq_str
        if freq in ['D', 'B']:
            max_lag = 30  # Monthly seasonality in daily data
        elif freq in ['W', 'W-SUN', 'W-MON']:
            max_lag = 52  # Yearly seasonality in weekly data
        elif freq in ['M', 'MS']:
            max_lag = 24  # 2-year seasonality in monthly data
        elif freq in ['Q', 'QS']:
            max_lag = 8   # 2-year seasonality in quarterly data
        elif freq in ['H']:
            max_lag = 48  # 2-day seasonality in hourly data
        else:
            max_lag = min(len(values) // 3, 365)  # Default
    
    max_lag = min(max_lag, len(values) // 3)  # Ensure lag isn't too large
    
    if max_lag < 4:  # Too short for reliable seasonality detection
        return False
        
    # Calculate ACF
    acf_values = acf(values, nlags=max_lag, fft=True)
    
    # Check for significant autocorrelations at seasonal lags
    threshold = 1.96 / np.sqrt(len(values))  # 95% confidence threshold
    
    # Check for common seasonal periods
    seasonal_lags = [7, 12, 24, 30, 52, 365]
    seasonal_lags = [lag for lag in seasonal_lags if lag < max_lag]
    
    for lag in seasonal_lags:
        if abs(acf_values[lag]) > threshold * 2:  # Using 2x threshold for stronger evidence
            return True
    
    return False

def check_trend(data):
    """
    Detects if the time series has a significant trend.
    
    Args:
        data (TimeSeries): Input time series
        
    Returns:
        bool: True if trend is detected
    """
    from scipy import stats
    import numpy as np
    
    values = data.values().flatten()
    
    # Linear regression for trend detection
    x = np.arange(len(values))
    slope, _, _, p_value, _ = stats.linregress(x, values)
    
    # Return True if trend is statistically significant
    return p_value < 0.05 and abs(slope) > 0.01 * np.std(values)

def check_stationarity(data):
    """
    Tests for stationarity using Augmented Dickey-Fuller test.
    
    Args:
        data (TimeSeries): Input time series
        
    Returns:
        bool: True if data is stationary
    """
    from statsmodels.tsa.stattools import adfuller
    
    values = data.values().flatten()
    
    # ADF test
    result = adfuller(values, regression='c')
    p_value = result[1]
    
    # Return True if stationary (reject null hypothesis of unit root)
    return p_value < 0.05

def check_complexity(data):
    """
    Measures the complexity of the time series using sample entropy.
    
    Args:
        data (TimeSeries): Input time series
        
    Returns:
        float: Complexity score between 0 and 1
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    values = data.values().flatten()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    
    # Calculate zero crossings (simple complexity measure)
    zero_crossings = np.sum(np.diff(np.signbit(scaled_values)))
    max_possible = len(values) - 1
    
    # Approximate entropy through zero crossings rate
    complexity_score = min(1.0, zero_crossings / (max_possible * 0.5))
    
    return complexity_score

def check_missing_values(data):
    """
    Calculates the proportion of missing values in the time series.
    
    Args:
        data (TimeSeries): Input time series
        
    Returns:
        float: Proportion of missing values (0 to 1)
    """
    import numpy as np
    
    values = data.values()
    missing_prop = np.isnan(values).mean()
    
    return missing_prop

def forecast_with_ets(data, forecast_steps=10):
    """
    Creates forecasts using Exponential Smoothing (ETS).
    
    Args:
        data (TimeSeries): Input time series in darts TimeSeries format
        forecast_steps (int): Number of steps to forecast
    
    Returns:
        pd.DataFrame: Forecast results with datetime index
    """
    from darts.models import ExponentialSmoothing
    
    # Initialize and fit the model
    model = ExponentialSmoothing(seasonal_periods=infer_seasonality(data))
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def forecast_with_prophet(data, forecast_steps=10):
    """
    Creates forecasts using Facebook Prophet.
    
    Args:
        data (TimeSeries): Input time series in darts TimeSeries format
        forecast_steps (int): Number of steps to forecast
    
    Returns:
        pd.DataFrame: Forecast results with datetime index
    """
    from darts.models import Prophet
    
    # Initialize and fit the model
    model = Prophet(
        yearly_seasonality=infer_yearly_seasonality(data),
        weekly_seasonality=infer_weekly_seasonality(data),
        daily_seasonality=infer_daily_seasonality(data)
    )
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def forecast_with_autoarima(data, forecast_steps=10):
    """
    Creates forecasts using Auto ARIMA.
    
    Args:
        data (TimeSeries): Input time series in darts TimeSeries format
        forecast_steps (int): Number of steps to forecast
    
    Returns:
        pd.DataFrame: Forecast results with datetime index
    """
    from darts.models import AutoARIMA
    
    # Initialize and fit the model
    model = AutoARIMA(seasonal=True, suppress_warnings=True)
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def forecast_with_lightgbm(data, forecast_steps=10):
    """
    Creates forecasts using LightGBM with advanced features.
    
    Args:
        data (TimeSeries): Input time series in darts TimeSeries format
        forecast_steps (int): Number of steps to forecast
    
    Returns:
        pd.DataFrame: Forecast results with datetime index
    """
    from darts.models import LightGBMModel
    from darts.dataprocessing.transformers import Scaler
    
    # Scale the data
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data)
    
    # Generate time features
    model = LightGBMModel(
        lags=[-1, -2, -3, -7, -14],  # Use multiple past values
        lags_past_covariates=[0, 1, 2, 3],  # Use covariates at current and future steps
        output_chunk_length=forecast_steps,
        add_encoders={
            "cyclic": {"future": ["month", "day", "dayofweek", "hour"]},
            "datetime_attribute": {
                "future": [
                    "month", "day", "dayofweek", "hour", "dayofyear",
                    "week", "quarter"
                ]
            }
        }
    )
    
    # Fit model
    model.fit(scaled_data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Inverse transform to get original scale
    forecast = scaler.inverse_transform(forecast)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def infer_seasonality(data):
    """Infers the seasonal period from the data frequency"""
    freq = data.freq_str
    
    if freq in ['D', 'B']:
        return 7  # Weekly seasonality for daily data
    elif freq in ['H']:
        return 24  # Daily seasonality for hourly data
    elif freq in ['M', 'MS']:
        return 12  # Yearly seasonality for monthly data
    elif freq in ['Q', 'QS']:
        return 4   # Yearly seasonality for quarterly data
    else:
        return None  # No seasonality

def infer_yearly_seasonality(data):
    """Determines if yearly seasonality should be included"""
    freq = data.freq_str
    return freq in ['D', 'B', 'W', 'M', 'MS', 'Q', 'QS']

def infer_weekly_seasonality(data):
    """Determines if weekly seasonality should be included"""
    freq = data.freq_str
    return freq in ['D', 'B', 'H']

def infer_daily_seasonality(data):
    """Determines if daily seasonality should be included"""
    freq = data.freq_str
    return freq in ['H', 'min']