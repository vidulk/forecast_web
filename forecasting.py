import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive, AutoTheta
from mlforecast import MLForecast
import lightgbm as lgb # <--- Import a model like LightGBM
from sklearn.metrics import mean_absolute_percentage_error
from mlforecast.auto import AutoMLForecast

def get_stats_forecasting_model(data, season_length=7, granularity=1, force_arima=False):
    """
    Selects and returns an appropriate StatsForecast model object.
    (This is your renamed get_forecasting_model function)
    """
    n = len(data)
    
    # If covariates are present, ARIMAX is a strong choice.
    # If force_arima is true, we select it directly.
    if force_arima:
        models = [AutoARIMA(season_length=season_length)]
    elif n < 100:
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

def select_forecasting_model(data, domain=None, season_length=7, granularity='D', has_covariates=False):
    """
    Main model selection logic. Chooses between statistical and ML models.
    If covariates are present, it defaults to AutoARIMA.
    """
    # If the user has explicitly chosen to use covariates, always select a model that supports them.
    if has_covariates:
        print("Covariates enabled by user. Selecting StatsForecast model (AutoARIMAX).")
        model_object = get_stats_forecasting_model(data, season_length=season_length, granularity=granularity, force_arima=True)
        model_name = "AutoARIMA"  # It acts as ARIMAX with exogenous features
        return model_object, model_name

    # Fallback to original logic if covariates are not explicitly enabled.
    if domain == 'retail' and len(data) >= 50:
        print("Selecting MLForecast model.")
        model_object = get_ml_forecasting_model(data, season_length=season_length, granularity=granularity)
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

def calculate_cv_accuracy(data=None, model_object=None, forecast_horizon=10, num_folds=1, stride=1, cv_df=None, model_name_col=None):
    """
    Calculates cross-validation accuracy.
    Can either run cross-validation itself or use a pre-computed CV dataframe.
    """
    # If a pre-computed cross-validation dataframe is not provided, run it.
    if cv_df is None:
        if data is None or model_object is None:
            raise ValueError("Must provide either cv_df or (data and model_object).")
        
        cv_df = model_object.cross_validation(
            df=data,
            h=forecast_horizon,
            step_size=stride,
            n_windows=num_folds
        )

    if cv_df.empty:
        return {'mape': None}

    # Determine the name of the forecast column if not provided.
    if model_name_col is None:
        if isinstance(model_object, StatsForecast):
            model_name_col = model_object.models[0].__class__.__name__
        elif isinstance(model_object, (MLForecast, AutoMLForecast)):
            model_name_col = model_object.models[0].__class__.__name__
        else:
            # Fallback if we can't determine the model name
            return {'mape': None}

    cv_df.dropna(inplace=True)
    if cv_df.empty or model_name_col not in cv_df.columns:
        # If the auto-detected name is wrong (e.g., 'AutoARIMAX'), try the explicit one.
        if model_name_col == 'AutoARIMA' and 'AutoARIMAX' in cv_df.columns:
            model_name_col = 'AutoARIMAX'
        else:
            return {'mape': None}

    sum_abs_error = (cv_df[model_name_col] - cv_df['y']).abs().sum()
    sum_actuals = cv_df['y'].abs().sum()

    if sum_actuals == 0:
        return {'mape': float('inf')} # Avoid division by zero

    mape = sum_abs_error / sum_actuals
    return {'mape': mape}


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
