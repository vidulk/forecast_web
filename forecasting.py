import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive, AutoTheta
from mlforecast import MLForecast
import lightgbm as lgb # <--- Import a model like LightGBM
from sklearn.metrics import mean_absolute_percentage_error

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
    data = data[['ds', 'y']].copy()
    data['unique_id'] = 0
    data = data[['unique_id', 'ds', 'y']]
    return data

def get_stats_forecasting_model(data):
    """
    Selects and returns an appropriate StatsForecast model object.
    (This is your renamed get_forecasting_model function)
    """
    n = len(data)
    if n < 30:
        models = [SeasonalNaive(season_length=7)]
    elif n < 100:
        models = [AutoETS(season_length=7)]
    else:
        models = [AutoARIMA()]
    
    model = StatsForecast(models=models, freq='D', n_jobs=1)
    return model

def get_ml_forecasting_model(data):
    """
    Creates and returns an MLForecast model object.
    """
    # Define the ML models you want to use (e.g., LightGBM)
    ml_models = [lgb.LGBMRegressor(random_state=0, n_estimators=100)]
    
    # Create the MLForecast object, defining features like lags and date features
    model = MLForecast(
        models=ml_models,
        freq='D',
        lags=[7, 14, 21],
        date_features=['dayofweek', 'month', 'year'],
    )
    return model

def select_forecasting_model(data, domain=None, return_name=True):
    """
    Main model selection logic. Chooses between statistical and ML models.
    """
    # Example logic: use ML for 'retail' domain, otherwise use statistical
    if domain == 'retail' and len(data) >= 50:
        print("Selecting MLForecast model.")
        model_object = get_ml_forecasting_model(data)
        # The model name is the class name of the first model in the list
        model_name = model_object.models[0].__class__.__name__
    else:
        print("Selecting StatsForecast model.")
        model_object = get_stats_forecasting_model(data)
        model_name = model_object.models[0].__class__.__name__

    return model_object, model_name

# --- REFACTORED BASELINE LOGIC ---

def get_baseline_model_object():
    """
    Creates and returns a StatsForecast object for the baseline model.
    This function now returns the object instead of a forecast.
    """
    model = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=1)
    return model

def calculate_cv_accuracy(data, model_object, forecast_horizon=10, num_folds=3, stride=1):
    """
    Calculates cross-validation accuracy. This function now works for
    both StatsForecast and MLForecast objects.
    """
    cv_df = model_object.cross_validation(
        df=data,
        h=forecast_horizon,
        step_size=stride,
        n_windows=num_folds
    )

    if cv_df.empty:
        return {'mape': None}

    # Get the model name to find the forecast column
    # This logic needs to be robust for both library types
    if isinstance(model_object, StatsForecast):
        model_name_col = model_object.models[0].__class__.__name__
    elif isinstance(model_object, MLForecast):
        # For MLForecast, the column name is the model's class name directly
        model_name_col = model_object.models[0].__class__.__name__
    else:
        # Fallback if something else is passed
        return {'mape': None}

    cv_df.dropna(inplace=True)
    if cv_df.empty or model_name_col not in cv_df.columns:
        return {'mape': None}

    mape = mean_absolute_percentage_error(cv_df['y'], cv_df[model_name_col])
    return {'mape': mape}

def convert_date_format(user_format):
    """Convert user-friendly date format to Python's date format"""
    # Common replacements
    replacements = {
        'DD': '%d',
        'MM': '%m',
        'YYYY': '%Y',
        'YY': '%y',
        'MON': '%b',
        'MONTH': '%B'
    }
    
    python_format = user_format
    for user_code, python_code in replacements.items():
        python_format = python_format.replace(user_code, python_code)
    
    return python_format
