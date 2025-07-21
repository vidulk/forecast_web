import os
import io
import uuid
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request, redirect, url_for, send_file, after_this_request, Response, jsonify
from forecasting import select_forecasting_model, calculate_cv_accuracy, prepare_data, convert_date_format, _generate_baseline_and_plot
import re
import boto3
from botocore.exceptions import ClientError
import time
import functools

# Set up Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size

# Timing decorator for debugging performance
def time_function(func_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            print(f"\n⏱️  [TIMER] Starting {name}...")
            app.logger.info(f"[TIMER] Starting {name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                print(f"✅ [TIMER] {name} completed in {duration:.2f} seconds")
                app.logger.info(f"[TIMER] {name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"❌ [TIMER] {name} failed after {duration:.2f} seconds: {str(e)}")
                app.logger.error(f"[TIMER] {name} failed after {duration:.2f} seconds: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Initialize S3 client
def get_s3_client():
    """Get S3 client with credentials from environment variables"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION')
    )

# S3 bucket name from environment variable
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')

# Helper functions for S3
def upload_to_s3(file_obj, key):
    """Upload a file object to S3"""
    s3 = get_s3_client()
    try:
        s3.upload_fileobj(file_obj, S3_BUCKET, key)
        return True
    except ClientError as e:
        print(f"[ERROR] S3 upload failed: {e}")
        return False

def read_from_s3(key, as_dataframe=True):
    """Read a file from S3, optionally return as DataFrame"""
    s3 = get_s3_client()
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        if as_dataframe:
            # Detect file type from key
            if key.endswith('.csv'):
                return pd.read_csv(io.BytesIO(response['Body'].read()))
            elif key.endswith('.xlsx'):
                return pd.read_excel(io.BytesIO(response['Body'].read()), engine='openpyxl')
            elif key.endswith('.xls'):
                return pd.read_excel(io.BytesIO(response['Body'].read()), engine='xlrd')
        else:
            return response['Body'].read()
    except ClientError as e:
        print(f"[ERROR] S3 read failed: {e}")
        return None

def save_dataframe_to_s3(df, key):
    """Save a DataFrame to S3 as CSV"""
    s3 = get_s3_client()
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        return True
    except ClientError as e:
        print(f"[ERROR] S3 dataframe save failed: {e}")
        return False

def delete_from_s3(key):
    """Delete a file from S3"""
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        print(f"[ERROR] S3 delete failed: {e}")
        return False

def generate_presigned_url(key, expiration=3600):
    """Generate a pre-signed URL for a file in S3"""
    s3 = get_s3_client()
    try:
        response = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': key},
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        print(f"[ERROR] Failed to generate presigned URL: {e}")
        return None

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

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected", 400

    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.csv', '.xlsx', '.xls'}:
        return "File type not allowed. Please upload CSV or Excel files.", 400

    try:
        if ext == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
        if df.empty:
            return "The uploaded file doesn't contain any data", 400
        temp_filename = f"temp_{uuid.uuid4().hex}.csv"
        if not save_dataframe_to_s3(df, temp_filename):
            return "Error saving file to storage", 500
    except Exception as e:
        print(f"[ERROR] File processing error: {e}")
        return f"Error processing file: {e}", 400

    return redirect(url_for('configure', filename=temp_filename))

@app.route('/configure/<filename>')
def configure(filename):
    # Try to detect date format and granularity from the file
    detected_format = "DD/MM/YYYY"  # Default
    detected_granularity = "daily" # Default
    
    try:
        # Read sample from S3
        df_sample = read_from_s3(filename)
        if df_sample is not None:
            detected_format = detect_date_format(df_sample)
            detected_granularity = detect_granularity(df_sample.copy())
            print(f"[DEBUG] Detected date format: {detected_format}")
            print(f"[DEBUG] Detected granularity: {detected_granularity}")

    except Exception as e:
        print(f"[WARNING] Could not detect date format or granularity: {e}")
    
    return render_template('configure.html', 
                           filename=filename, 
                           detected_format=detected_format,
                           detected_granularity=detected_granularity)

@app.route('/process', methods=['POST'])
@time_function("PROCESS_ROUTE")
def process():
    filename = request.form.get('filename')
    date_column = request.form.get('date_column', 'ds')
    date_format = request.form.get('date_format', 'DD/MM/YYYY')
    value_column = request.form.get('value_column', 'value')
    steps = request.form.get('steps', type=int, default=7)
    granularity = request.form.get('granularity', 'daily')
    
    if not steps or steps <= 0:
        return "Invalid timesteps value", 400
    
    # Time the S3 read operation
    start_time = time.time()
    print(f"\n⏱️  [TIMER] Reading file from S3...")
    try:
        df = read_from_s3(filename)
        if df is None:
            return "Error reading file from storage", 500
    except Exception as e:
        return f"Error reading file: {str(e)}", 400
    
    read_time = time.time() - start_time
    print(f"✅ [TIMER] S3 file read completed in {read_time:.2f} seconds")
        
    # Time the data processing
    start_time = time.time()
    print(f"⏱️  [TIMER] Processing data (rename columns, convert dates)...")
    
    # Rename columns to the expected formats
    if date_column != 'ds':
        df = df.rename(columns={date_column: 'ds'})
    if value_column != 'y':
        df = df.rename(columns={value_column: 'y'})
    
    # Convert date format
    python_date_format = convert_date_format(date_format)
    print(f"[DEBUG] Converted date format: {python_date_format}")
    
    # Special handling for Year-Month format
    if date_format == "YYYY-MM":
        df['ds'] = pd.to_datetime(df['ds'].astype(str) + '-01', errors='coerce', format=python_date_format + '-%d')
    else:
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce', format=python_date_format)
    
    # Check if date conversion was successful
    if df['ds'].isnull().any():
        app.logger.error(f"Date conversion failed with format: {date_format}. Resulting 'ds' column has nulls.")
        error_message = "The date format you selected doesn't seem to match your data. Please check the format and try again."
        return render_template('error.html', message=error_message, filename=filename), 400
    
    process_time = time.time() - start_time
    print(f"✅ [TIMER] Data processing completed in {process_time:.2f} seconds")
    
    # Time the S3 save operation
    start_time = time.time()
    print(f"⏱️  [TIMER] Saving processed file to S3...")
    
    processed_filename = f"processed_{filename}"
    if not save_dataframe_to_s3(df, processed_filename):
        return "Error saving processed file", 500
    
    save_time = time.time() - start_time
    print(f"✅ [TIMER] S3 save completed in {save_time:.2f} seconds")
    
    # Delete the original temporary file
    delete_from_s3(filename)
    
    # Preserve all original parameters for the next step
    query_params = request.form.to_dict()
    query_params.pop('filename', None) # filename is in the URL path
    
    return redirect(url_for('forecast_loading', filename=processed_filename, **query_params))

@app.route('/forecast_loading/<filename>')
def forecast_loading(filename):
    """Renders a loading page that will trigger the forecast generation."""
    return render_template('loading.html', filename=filename)

@app.route('/api/forecast/<filename>')
@time_function("FORECAST_API")
def forecast_api(filename):
    route_start_time = time.time()
    
    steps = request.args.get('steps', default=10, type=int)
    granularity_str = request.args.get('granularity', default='daily')
    domain = request.args.get('domain', default=None)

    # Map string granularity to frequency strings for Nixtla
    granularity_map = {
        'hourly': 'H',
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M'
    }
    granularity = granularity_map.get(granularity_str, 'D')

    # Determine seasonality based on granularity
    if granularity_str == 'hourly':
        seasonality = 24
    elif granularity_str == 'daily':
        seasonality = 7
    elif granularity_str == 'weekly':
        seasonality = 52
    elif granularity_str == 'monthly':
        seasonality = 12
    else:
        seasonality = 7

    app.logger.info(f"Starting forecast for file: {filename}, steps: {steps}, granularity: {granularity}, seasonality: {seasonality}, domain: {domain}")
    
    try:
        # Time data preparation
        prepped_data = _prepare_data_from_s3(filename)
        if prepped_data is None:
            return jsonify({'error': "Error reading processed file from storage"}), 500
        
        forecast_df, expected_accuracy, model_name = _generate_forecast_and_accuracy(prepped_data, steps, domain, season_length=seasonality, granularity=granularity)            
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')  # Ensure date is in YYYY-MM-DD format
        # Time plot generation
        start_time = time.time()
        print(f"⏱️  [TIMER] Generating baseline and plot...")
        plot_html, baseline_accuracy = _generate_baseline_and_plot(prepped_data, steps, forecast_df, granularity)
        plot_time = time.time() - start_time
        print(f"✅ [TIMER] Plot generation completed in {plot_time:.2f} seconds")
        
        # Time forecast save
        start_time = time.time()
        print(f"⏱️  [TIMER] Saving forecast to S3...")
        forecast_filename = f"forecast_{filename}"
        if not save_dataframe_to_s3(forecast_df, forecast_filename):
            return jsonify({'error': "Error saving forecast file"}), 500
        save_time = time.time() - start_time
        print(f"✅ [TIMER] Forecast save completed in {save_time:.2f} seconds")

        # Save plot to S3 to be retrieved by the result page
        plot_filename = f"plot_{filename}.html"
        s3 = get_s3_client()
        s3.put_object(Bucket=S3_BUCKET, Key=plot_filename, Body=plot_html.encode('utf-8'))
        
        # Print total route timing
        total_route_time = time.time() - route_start_time
        print(f"\n🎯 [TIMER] TOTAL FORECAST API TIME: {total_route_time:.2f} seconds")

        result_url = url_for('forecast_result', 
                             filename=filename, 
                             forecast_file=forecast_filename,
                             plot_file=plot_filename,
                             expected_accuracy=expected_accuracy,
                             baseline_accuracy=baseline_accuracy)

        return jsonify({'result_url': result_url})
        
    except Exception as e:
        app.logger.error(f"Exception during forecasting API call: {str(e)}", exc_info=True)
        return jsonify({'error': f"Error during forecasting: {str(e)}"}), 500

@app.route('/forecast/result/<filename>')
def forecast_result(filename):
    """Displays the final forecast result."""
    try:
        plot_file = request.args.get('plot_file')
        forecast_file = request.args.get('forecast_file')
        expected_accuracy = request.args.get('expected_accuracy')
        baseline_accuracy = request.args.get('baseline_accuracy')

        # Read plot from S3
        plot_html_bytes = read_from_s3(plot_file, as_dataframe=False)
        if plot_html_bytes is None:
            return "Error loading plot.", 500
        plot_html = plot_html_bytes.decode('utf-8')

        @after_this_request
        def cleanup(response):
            try:
                # Clean up the processed data file and the plot file
                delete_from_s3(filename)
                delete_from_s3(plot_file)
                print(f"[CLEANUP] Removed processed file: {filename} and plot file: {plot_file}")
            except Exception as e:
                print(f"[ERROR] Failed to clean up files: {e}")
            return response

        return render_template('result.html', 
                               plot_html=plot_html, 
                               forecast_file=forecast_file, 
                               expected_accuracy=expected_accuracy, 
                               baseline_accuracy=baseline_accuracy)
    except Exception as e:
        app.logger.error(f"Error rendering forecast result: {str(e)}", exc_info=True)
        return "Error displaying forecast result.", 500

@time_function("PREPARE_DATA_FROM_S3")
def _prepare_data_from_s3(filename):
    app.logger.info(f"Reading file from S3: {filename}")
    
    # Time the S3 read
    start_time = time.time()
    df = read_from_s3(filename)
    if df is None:
        app.logger.error(f"Error reading processed file - returned None: {filename}")
        return None
    read_time = time.time() - start_time
    print(f"⏱️  S3 read took {read_time:.2f} seconds")
    
    app.logger.info(f"Successfully read file, shape: {df.shape}")
    
    # Time the data cleaning
    start_time = time.time()
    app.logger.info("Cleaning column names")
    df.columns = [str(c).strip().replace("=", "").replace("(", "").replace(")", "") for c in df.columns]
    app.logger.info(f"Columns after cleaning: {df.columns.tolist()}")
    clean_time = time.time() - start_time
    print(f"⏱️  Data cleaning took {clean_time:.2f} seconds")
    
    # Time the data preparation
    start_time = time.time()
    app.logger.info("Preparing data for forecasting")
    prepped_data = prepare_data(df.copy())
    prep_time = time.time() - start_time
    print(f"⏱️  Data preparation took {prep_time:.2f} seconds")
    
    app.logger.info(f"Data prepared, length: {len(prepped_data)}")
    return prepped_data

@time_function("GENERATE_FORECAST_AND_ACCURACY")
def _generate_forecast_and_accuracy(prepped_data, steps, domain, season_length=7, granularity='D'):
    # Time model selection
    start_time = time.time()
    print(f"⏱️  [TIMER] Selecting forecasting model...")
    model_object, model_name = select_forecasting_model(prepped_data, domain, season_length=season_length, granularity=granularity)
    model_selection_time = time.time() - start_time
    print(f"✅ [TIMER] Model selection completed in {model_selection_time:.2f} seconds")
    app.logger.info(f"Selected model: {model_name}")
    
    # Time model fitting
    start_time = time.time()
    print(f"⏱️  [TIMER] Fitting model and generating forecast...")
    app.logger.info("Fitting model and generating forecast...")
    model_object.fit(prepped_data)
    fit_time = time.time() - start_time
    print(f"✅ [TIMER] Model fitting completed in {fit_time:.2f} seconds")
    
    # Time prediction
    start_time = time.time()
    print(f"⏱️  [TIMER] Generating predictions...")
    forecast_df = model_object.predict(h=steps)
    forecast_df.rename(columns={model_name: 'y'}, inplace=True)
    predict_time = time.time() - start_time
    print(f"✅ [TIMER] Prediction generation completed in {predict_time:.2f} seconds")
    
    # Time cross-validation
    start_time = time.time()
    print(f"⏱️  [TIMER] Calculating cross-validation accuracy...")
    app.logger.info("Calculating cross-validation accuracy...")
    num_folds = 1
    app.logger.info(f"Using {num_folds} folds for cross-validation")
    accuracy_results = calculate_cv_accuracy(
        prepped_data,
        model_object,
        forecast_horizon=steps,
        num_folds=num_folds,
        stride=1
    )
    cv_time = time.time() - start_time
    print(f"✅ [TIMER] Cross-validation completed in {cv_time:.2f} seconds")
    
    expected_accuracy = round(accuracy_results['mape'] * 100, 3) if accuracy_results.get('mape') is not None else 'N/A'
    app.logger.info(f"Forecast generated, shape: {forecast_df.shape}")
    
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

@app.route('/download/<path:forecast_file>')
def download_file(forecast_file):
    try:
        # Read the file from S3
        file_data = read_from_s3(forecast_file, as_dataframe=False)
        if file_data is None:
            return "Error reading forecast file", 500
        
        # Set up cleanup to delete the forecast file after download
        @after_this_request
        def cleanup(response):
            try:
                delete_from_s3(forecast_file)
                print(f"[CLEANUP] Removed downloaded forecast file: {forecast_file}")
            except Exception as e:
                print(f"[ERROR] Failed to remove downloaded file: {e}")
            return response
        
        # Create a response with the file data
        response = Response(
            file_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={forecast_file.split("/")[-1]}'}
        )
        
        return response
        
    except Exception as e:
        
        print(f"[ERROR] Download error: {e}")
        return "Error preparing download", 500

# Add a scheduled cleanup function for S3 objects
def cleanup_old_s3_files():

    """Remove files older than 1 hour from S3 bucket""" 
    import time
    from datetime import datetime, timedelta
    
    s3 = get_s3_client()
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    try:
        objects = s3.list_objects_v2(Bucket=S3_BUCKET)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                # If object is older than 1 hour 
                if obj['LastModified'].replace(tzinfo=None) < one_hour_ago:
                    try:
                        s3.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
                        print(f"[CLEANUP] Removed old S3 file: {obj['Key']}")
                    except Exception as e:
                        print(f"[ERROR] Failed to remove old S3 file {obj['Key']}: {e}")
    except Exception as e:
        print(f"[ERROR] Error listing S3 objects: {e}")

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions and log them properly"""
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    
    cleanup_old_s3_files()  # Clean up old files on startup
    app.run(debug=True)
