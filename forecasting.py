import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive, AutoTheta
from mlforecast import MLForecast
import lightgbm as lgb # <--- Import a model like LightGBM
from sklearn.metrics import mean_absolute_percentage_error
from mlforecast.auto import AutoMLForecast

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

def get_stats_forecasting_model(data, season_length=7, granularity=1):
    """
    Selects and returns an appropriate StatsForecast model object.
    (This is your renamed get_forecasting_model function)
    """
    n = len(data)
    if n < 100:
        models = [AutoTheta(season_length=season_length)]
    else:
        models = [AutoARIMA(season_length=season_length)]

    model = StatsForecast(models=models, freq=granularity, n_jobs=1)
    return model

def get_ml_forecasting_model(data, season_length=7, granularity='D'):
    """
    Creates and returns an MLForecast model object.
    """

    ml_models = [lgb.LGBMRegressor(random_state=0, n_estimators=100)]
    
    model = AutoMLForecast(
        models=ml_models,
        freq=granularity,
        season_length=season_length,
        n_jobs=1
    )
        
    return model

def select_forecasting_model(data, domain=None, season_length=7, granularity='D'):
    """
    Main model selection logic. Chooses between statistical and ML models.
    """
    # Example logic: use ML for 'retail' domain, otherwise use statistical
    if domain == 'retail' and len(data) >= 50:
        print("Selecting MLForecast model.")
        model_object = get_ml_forecasting_model(data, season_length=season_length, granularity=granularity)
        # The model name is the class name of the first model in the list
        model_name = model_object.models[0].__class__.__name__
    else:
        print("Selecting StatsForecast model.")
        model_object = get_stats_forecasting_model(data, season_length=season_length, granularity=granularity)
        model_name = model_object.models[0].__class__.__name__

    return model_object, model_name

def get_baseline_model_object(granularity='D'):
    """
    Creates and returns a StatsForecast object for the baseline model.
    This function now returns the object instead of a forecast.
    """
    model = StatsForecast(models=[SeasonalNaive(season_length=7)], freq=granularity, n_jobs=1)
    return model

def calculate_cv_accuracy(data, model_object, forecast_horizon=10, num_folds=1, stride=1):
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
    elif isinstance(model_object, (MLForecast, AutoMLForecast)):
        # For MLForecast, the column name is the model's class name directly
        model_name_col = model_object.models[0].__class__.__name__
    else:
        # Fallback if something else is passed
        return {'mape': None}

    cv_df.dropna(inplace=True)
    if cv_df.empty or model_name_col not in cv_df.columns:
        return {'mape': None}

    sum_abs_error = (cv_df[model_name_col] - cv_df['y']).abs().sum()
    sum_actuals = cv_df['y'].abs().sum()

    if sum_actuals == 0:
        return {'mape': float('inf')} # Avoid division by zero

    mape = sum_abs_error / sum_actuals
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

import plotly.graph_objects as go

def _generate_baseline_and_plot(prepped_data, steps, forecast_df, granularity):
    """
    Generates a plot for the forecast, baseline, and original data.
    Also calculates the baseline accuracy.
    """
    # --- Calculate Baseline Forecast ---
    baseline_model_object = get_baseline_model_object(granularity=granularity)
    baseline_model_object.fit(prepped_data)
    base_forecast_df = baseline_model_object.predict(h=steps)
    base_forecast_df.rename(columns={'SeasonalNaive': 'value'}, inplace=True)
        
    # --- Create the Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prepped_data['ds'], y=prepped_data['y'], mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], mode='lines', name='Model Forecast', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=base_forecast_df['ds'], y=base_forecast_df['value'], mode='lines', name='Baseline Forecast', line=dict(dash='dot')))

    fig.update_layout(
        title_text='Forecast vs. Actuals',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
    )

    # --- Calculate Baseline Accuracy ---
    baseline_accuracy_results = calculate_cv_accuracy(
        data=prepped_data,
        model_object=baseline_model_object,
        forecast_horizon=steps,
        num_folds=1,
        stride=1
    )
    
    baseline_mape = baseline_accuracy_results.get('mape')
    baseline_accuracy = round(baseline_mape * 100, 3) if baseline_mape is not None else 'N/A'
    
    config = {'displayModeBar': False}
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False, config=config)
    return plot_html, baseline_accuracy
