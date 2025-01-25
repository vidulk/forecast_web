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
    
    if len(data) == 0:
        raise ValueError("DataFrame must contain at least one row of data.")
    
    if len(data) > 1000000:
        raise ValueError("DataFrame must contain less than 1,000,000 rows of data.")
    
    # Convert 'dt' to datetime and sort the data
    try:
        data['dt'] = pd.to_datetime(data['dt'], format='%d/%m/%y')
    except:
        pass

    # reformat dt column to be in YYYY-MM-DD format

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
    data = prepare_data(data)
    # Initialize and fit the model
    model = Theta()  
    model.fit(data)
    
    # Generate forecast
    forecast = model.predict(forecast_steps)
    
    # Convert forecast to DataFrame
    forecast_df = forecast.pd_dataframe().reset_index()
    forecast_df['dt'] = forecast_df['dt'].dt.strftime('%Y-%m-%d')
    
    return forecast_df

def baseline_forecast(data, forecast_steps=10):
    
    data = prepare_data(data)
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