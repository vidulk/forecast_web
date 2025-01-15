import pandas as pd
from darts import TimeSeries
from darts.models import Theta

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
    data['dt'] = pd.to_datetime(data['dt'], format='%d/%m/%y')
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

def get_accuracy(forecast_df, actual_df):

    from sklearn.model_selection import TimeSeriesSplit

    # calc cross validation accuracy
    n_splits = 5  
    tscv = TimeSeriesSplit(n_splits=n_splits)

    val_preds = []
    val_scores = []
    val_mapes = []
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categ_features)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categ_features)
        
        gbm = lgb.train(params, train_data, valid_sets=[val_data],)
        y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        
        y_pred[y_pred < 0] = 0     # Set negative predictions to 0
        
        # Calculate validation score
        val_smape = 100 * np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_val) + np.abs(y_pred)))
        try:
            mape_pred = np.where(y_pred == 0, 0.01, y_pred)
            mape_val = np.where(y_val == 0, 0.01, y_val)
            val_mape = 100 * np.mean(np.abs((mape_pred - mape_val) / mape_val))
        except ZeroDivisionError:
            val_mape = 0

        val_scores.append(val_smape)
        val_mapes.append(val_mape)
        # Store predictions
        val_preds.extend(y_pred)

    # Calculate the average validation score across folds
    avg_val_score = np.mean(val_scores)
    print("Average Validation SMAPE:", avg_val_score)

    avg_val_mape = np.mean(val_mapes)
    print("Average Validation MAPE:", avg_val_mape)
    
    return accuracy