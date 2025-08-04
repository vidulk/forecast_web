"""
Forecast service containing business logic for forecasting operations.
"""
import time
import pandas as pd
from flask import current_app
from forecasting import (
    select_forecasting_model, 
    calculate_cv_accuracy, 
    _generate_baseline_and_plot
)
from utils.timing import time_function
from utils.s3_client import read_from_s3, save_dataframe_to_s3
from utils.data_utils import prepare_data
from config import GRANULARITY_MAP, SEASONALITY_MAP


@time_function("PREPARE_DATA_FROM_S3")
def prepare_data_from_s3(filename, forecast_horizon, granularity, use_covariates):
    """Prepare data from S3 file for forecasting"""
    current_app.logger.info(f"Reading file from S3: {filename}")
    
    # Time the S3 read
    start_time = time.time()
    df = read_from_s3(filename)
    if df is None:
        current_app.logger.error(f"Error reading processed file - returned None: {filename}")
        return None, None, None
    read_time = time.time() - start_time
    print(f"⏱️  S3 read took {read_time:.2f} seconds")
    
    current_app.logger.info(f"Successfully read file, shape: {df.shape}")
    
    # Time the data cleaning
    start_time = time.time()
    current_app.logger.info("Cleaning column names")
    df.columns = [str(c).strip().replace("=", "").replace("(", "").replace(")", "") for c in df.columns]
    current_app.logger.info(f"Columns after cleaning: {df.columns.tolist()}")
    clean_time = time.time() - start_time
    print(f"⏱️  Data cleaning took {clean_time:.2f} seconds")

    # Logic to separate covariates and create future dataframe
    X_df = None
    futr_df = None
    if use_covariates:
        covariate_cols = [col for col in df.columns if col.startswith('x_')]
        if covariate_cols:
            # Ensure 'ds' is datetime for proper indexing
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Historical covariates
            X_df = df[['ds'] + covariate_cols].copy()
            
            # Create future dates for the forecast horizon
            last_date = df['ds'].max()
            future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=granularity)[1:]
            
            # Create the future covariates dataframe (futr_df)
            # For this implementation, we forward-fill the last known value.
            # A more advanced implementation might require users to upload future values.
            if not X_df.empty:
                last_known_covariates = X_df.iloc[-1:][covariate_cols]
                futr_df = pd.DataFrame({'ds': future_dates})
                for col in covariate_cols:
                    futr_df[col] = last_known_covariates[col].values[0]
                
                # Add unique_id to both dataframes for the model
                X_df['unique_id'] = 0
                futr_df['unique_id'] = 0

    # Time the data preparation
    start_time = time.time()
    current_app.logger.info("Preparing data for forecasting")
    prepped_data = prepare_data(df.copy())
    prep_time = time.time() - start_time
    print(f"⏱️  Data preparation took {prep_time:.2f} seconds")
    
    current_app.logger.info(f"Data prepared, length: {len(prepped_data)}")
    return prepped_data, X_df, futr_df


@time_function("GENERATE_FORECAST_AND_ACCURACY")
def generate_forecast_and_accuracy(prepped_data, steps, domain, season_length=7, granularity='D', has_covariates=False, X_df=None, futr_df=None):
    """Generate forecast and calculate accuracy"""
    # Time model selection
    start_time = time.time()
    print(f"⏱️  [TIMER] Selecting forecasting model...")
    model_object, model_name = select_forecasting_model(
        prepped_data, 
        domain, 
        season_length=season_length, 
        granularity=granularity,
        has_covariates=has_covariates
    )
    model_selection_time = time.time() - start_time
    print(f"✅ [TIMER] Model selection completed in {model_selection_time:.2f} seconds")
    current_app.logger.info(f"Selected model: {model_name}")
    
    # Time model fitting
    start_time = time.time()
    print(f"⏱️  [TIMER] Fitting model and generating forecast...")
    current_app.logger.info("Fitting model and generating forecast...")
    # The `fit` method uses the historical covariates already present in `prepped_data`.
    model_object.fit(prepped_data)
    fit_time = time.time() - start_time
    print(f"✅ [TIMER] Model fitting completed in {fit_time:.2f} seconds")
    
    # Time prediction
    start_time = time.time()
    print(f"⏱️  [TIMER] Generating predictions...")
    # Pass FUTURE covariates to the predict method if they exist.
    forecast_df = model_object.predict(h=steps, X_df=futr_df)
    forecast_df.rename(columns={model_name: 'y'}, inplace=True)
    predict_time = time.time() - start_time
    print(f"✅ [TIMER] Prediction generation completed in {predict_time:.2f} seconds")
    
    # Time cross-validation
    start_time = time.time()
    print(f"⏱️  [TIMER] Calculating cross-validation accuracy...")
    current_app.logger.info("Calculating cross-validation accuracy...")
    num_folds = 1
    current_app.logger.info(f"Using {num_folds} folds for cross-validation")
    
    # Cross-validation needs access to historical covariates
    cv_df = model_object.cross_validation(
        df=prepped_data,
        h=steps,
        step_size=steps, # A common setting to avoid overlapping windows
        n_windows=num_folds,
    )

    # This part of accuracy calculation might need adjustment if CV results change with covariates
    if not cv_df.empty:
        accuracy_results = calculate_cv_accuracy(
            cv_df=cv_df, # Pass the results dataframe
            model_name_col=model_name
        )
    else:
        accuracy_results = {'mape': None}

    cv_time = time.time() - start_time
    print(f"✅ [TIMER] Cross-validation completed in {cv_time:.2f} seconds")
    
    expected_accuracy = round(accuracy_results['mape'] * 100, 3) if accuracy_results.get('mape') is not None else 'N/A'
    current_app.logger.info(f"Forecast generated, shape: {forecast_df.shape}")
    
    # Print summary timing
    total_time = model_selection_time + fit_time + predict_time + cv_time
    print(f"\n📊 [TIMER SUMMARY] Forecast generation breakdown:")
    print(f"   Model Selection: {model_selection_time:.2f}s ({model_selection_time/total_time*100:.1f}%)")
    print(f"   Model Fitting:   {fit_time:.2f}s ({fit_time/total_time*100:.1f}%)")
    print(f"   Prediction:      {predict_time:.2f}s ({predict_time/total_time*100:.1f}%)")
    print(f"   Cross-validation:{cv_time:.2f}s ({cv_time/total_time*100:.1f}%)")
    print(f"   TOTAL:           {total_time:.2f}s")
    
    print(f"\n[DEBUG] Selected model: {model_name}")
    print(f"[DEBUG] Data length: {len(prepped_data)}")
    print(f"[DEBUG] Domain: {domain if domain else 'Not specified'}")
    print(f"[DEBUG] Forecast steps: {steps}\n")
    return forecast_df, expected_accuracy, model_name


def process_forecast_request(filename, steps, granularity_str, domain, use_covariates):
    """
    Main function to process a forecast request.
    
    Args:
        filename: S3 filename
        steps: Number of forecast steps
        granularity_str: Granularity string ('daily', 'weekly', etc.)
        domain: Domain context
        use_covariates: Whether to use covariates
        
    Returns:
        tuple: (forecast_df, expected_accuracy, model_name, plot_html, baseline_accuracy)
    """
    # Map string granularity to frequency strings for Nixtla
    granularity = GRANULARITY_MAP.get(granularity_str, 'D')
    seasonality = SEASONALITY_MAP.get(granularity_str, 7)

    current_app.logger.info(f"Starting forecast for file: {filename}, steps: {steps}, granularity: {granularity}, seasonality: {seasonality}, domain: {domain}, use_covariates: {use_covariates}")
    
    # Prepare data
    prepped_data, X_df, futr_df = prepare_data_from_s3(filename, steps, granularity, use_covariates)
    if prepped_data is None:
        raise Exception("Error reading processed file from storage")
    
    # Generate forecast
    forecast_df, expected_accuracy, model_name = generate_forecast_and_accuracy(
        prepped_data, 
        steps, 
        domain, 
        season_length=seasonality, 
        granularity=granularity,
        has_covariates=use_covariates,
        X_df=X_df,
        futr_df=futr_df
    )
    
    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')  # Ensure date is in YYYY-MM-DD format
    
    # Generate plot
    start_time = time.time()
    print(f"⏱️  [TIMER] Generating baseline and plot...")
    plot_html, baseline_accuracy = _generate_baseline_and_plot(prepped_data, steps, forecast_df, granularity)
    plot_time = time.time() - start_time
    print(f"✅ [TIMER] Plot generation completed in {plot_time:.2f} seconds")
    
    return forecast_df, expected_accuracy, model_name, plot_html, baseline_accuracy
