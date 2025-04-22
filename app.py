import os
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request, redirect, url_for, send_file
from forecasting import select_forecasting_model, baseline_forecast, calculate_cv_accuracy, prepare_data, convert_date_format
import re

# In app.py, add this after creating the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'forecasts'
# Set maximum file size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB in bytes

# Ensure forecasts folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_date_format(df, date_column='dt'):
    """
    Attempts to detect the date format from the first few rows of data
    Returns a user-friendly format like DD/MM/YYYY
    """
    if date_column not in df.columns:
        # Try common date column names
        date_cols = ['date', 'time', 'datetime', 'timestamp', 'day']
        for col in date_cols:
            if col in df.columns:
                date_column = col
                break
        else:
            # If no date column found
            return "DD/MM/YYYY"  # Default format
    print(date_column)
    print(f"[DEBUG] Detecting date format for column: {date_column}")
    # Get a sample date value (first non-null)
    sample_dates = df[date_column].dropna()
    if len(sample_dates) == 0:
        return "DD/MM/YYYY"  # Default if no valid dates
        
    sample = str(sample_dates.iloc[0])
    # Simple pattern detection based on common formats
    if re.match(r'\d{4}-\d{2}-\d{2}', sample):
        return "YYYY-MM-DD"
    elif re.match(r'\d{2}/\d{2}/\d{4}', sample):
        # Could be MM/DD/YYYY or DD/MM/YYYY
        # Try to distinguish based on the values
        parts = sample.split('/')
        if int(parts[0]) > 12:  # First part is > 12, likely a day
            return "DD/MM/YYYY"
        else:  # Could be month first
            return "MM/DD/YYYY"
    elif re.match(r'\d{2}/\d{2}/\d{2}', sample):
        return "DD/MM/YY"  # or MM/DD/YY, using same logic as above
    elif re.match(r'\d{2}-\w{3}-\d{4}', sample):
        return "DD-MON-YYYY"
    elif re.match(r'\w{3} \d{2}, \d{4}', sample):
        return "MON DD, YYYY"
    else:
        return "DD/MM/YYYY"  # Default format
    
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Check if the file extension is allowed
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return "File type not allowed. Please upload CSV or Excel files.", 400
    
    # Save dataframe to session or temporary file with a random name
    import uuid
    temp_filename = f"temp_{uuid.uuid4().hex}{file_ext}"
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    
    try:
        # First save the file temporarily to disk
        file.save(temp_filepath)
        
        # Now read it using the appropriate method
        if file_ext == '.csv':
            df = pd.read_csv(temp_filepath)
        elif file_ext == '.xlsx':
            df = pd.read_excel(temp_filepath, engine='openpyxl')
        elif file_ext == '.xls':
            df = pd.read_excel(temp_filepath, engine='xlrd')
        else:
            return "Unsupported file format", 400
        
        # Validate that the file contains data
        if len(df) == 0:
            return "The uploaded file doesn't contain any data", 400
        
        # Display first few rows for debugging
        print(f"[DEBUG] File type: {file_ext}, First few rows:")
        
        # Save processed dataframe back to CSV
        csv_temp_filename = f"temp_{uuid.uuid4().hex}.csv"
        csv_temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_temp_filename)
        df.to_csv(csv_temp_filepath, index=False)
        
        # Remove the original temporary file if it's not a CSV
        if file_ext != '.csv':
            os.remove(temp_filepath)
        
        # Use the CSV version for further processing
        temp_filename = csv_temp_filename
        
    except Exception as e:
        print(f"[ERROR] File processing error: {str(e)}")
        # Try to remove temporary file if it exists
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return f"Error processing file: {str(e)}", 400
    
    # Redirect to the config page with the temporary filename
    return redirect(url_for('configure', filename=temp_filename))

@app.route('/configure/<filename>')
def configure(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Try to detect date format from the file
    detected_format = "DD/MM/YYYY"  # Default
    try:
        # Read sample of the file (first 100 rows)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.csv':
            df_sample = pd.read_csv(filepath, nrows=100)
        elif file_ext in ['.xlsx', '.xls']:
            df_sample = pd.read_excel(filepath, nrows=100)
            
        detected_format = detect_date_format(df_sample)
        
    except Exception as e:
        print(f"[WARNING] Could not detect date format: {e}")
    
    return render_template('configure.html', filename=filename, detected_format=detected_format)

@app.route('/process', methods=['POST'])
def process():
    filename = request.form.get('filename')
    date_column = request.form.get('date_column', 'dt')
    date_format = request.form.get('date_format', 'DD/MM/YYYY')
    value_column = request.form.get('value_column', 'value')
    steps = request.form.get('steps', type=int, default=7)
    
    if not steps or steps <= 0:
        return "Invalid timesteps value", 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Read the file based on its extension
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            return "Unsupported file format", 400
    except Exception as e:
        return f"Error reading file: {str(e)}", 400
    
    print(date_column, date_format, value_column)
    # Rename columns to the expected format
    if date_column != 'dt':
        df = df.rename(columns={date_column: 'dt'})
    if value_column != 'value':
        df = df.rename(columns={value_column: 'value'})
    
    # Convert date format
    python_date_format = convert_date_format(date_format)
    print(f"\n[DEBUG] Python date format: {python_date_format}")
    try:
        df['dt'] = pd.to_datetime(df['dt'], format=python_date_format)
    except ValueError as e:
        return f"Could you go back and check your date format and make sure its correct?", 400
    
    # Save the processed file
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
    df.to_csv(processed_filepath, index=False)
    
    # Now redirect to forecast with the processed file
    return redirect(url_for('forecast', filename=f'processed_{filename}', steps=steps))

@app.route('/forecast/<filename>')
def forecast(filename):
    steps = request.args.get('steps', default=10, type=int)  # Default to 10 if not specified
    domain = request.args.get('domain', default=None)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df.columns = [str(c).strip().replace("=", "").replace("(", "").replace(")", "") for c in df.columns]
    
    prepped_data = prepare_data(df.copy())
    forecast_function, model_name = select_forecasting_model(prepped_data, domain, return_name=True)
    forecast_df = forecast_function(prepped_data, forecast_steps=steps)
    
    print(f"\n[DEBUG] Selected model: {model_name}")
    print(f"[DEBUG] Data length: {len(prepped_data)}")
    print(f"[DEBUG] Domain: {domain if domain else 'Not specified'}")
    print(f"[DEBUG] Forecast steps: {steps}\n")

    # Create an interactive Plo
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['dt'], y=df['value'], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=forecast_df['dt'], 
                             y=forecast_df['value'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title='Forecast vs Original', xaxis_title='Index', yaxis_title='Value')
    
    # predicted score on future data
    base_forecast = baseline_forecast(prepped_data, forecast_steps=steps)
    fig.add_trace(go.Scatter(x=base_forecast['dt'], 
                             y=base_forecast['value'], mode='lines', name='Baseline Forecast', line=dict(dash='dash')))

    # Add config to hide the mode bar
    config = {
        'displayModeBar': False,  # Hides the toolbar completely
    }
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False, config=config)

    forecast_path = os.path.join(app.config['UPLOAD_FOLDER'], f'forecast_{filename}')
    forecast_df.to_csv(forecast_path, index=False)
    
    expected_accuracy = round(calculate_cv_accuracy(
        prepped_data,
        forecast_function,
        forecast_horizon=steps,
        num_folds=2,
        stride=1
    )['mape'],3)

    baseline_accuracy = round(calculate_cv_accuracy(
        prepped_data,
        baseline_forecast,
        forecast_horizon=steps,
        num_folds=2,
        stride=1
    )['mape'],3)

    return render_template('result.html', plot_html=plot_html, forecast_file=forecast_path, expected_accuracy=expected_accuracy, baseline_accuracy=baseline_accuracy)

@app.route('/download/<path:forecast_file>')
def download_file(forecast_file):
    return send_file(forecast_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

