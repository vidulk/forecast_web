import os
import io
import uuid
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, render_template, request, redirect, url_for, send_file, after_this_request, Response
from forecasting import select_forecasting_model, baseline_forecast, calculate_cv_accuracy, prepare_data, convert_date_format
import re
import boto3
from botocore.exceptions import ClientError

# Set up Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size

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

# Keep the existing detect_date_format function
def detect_date_format(df, date_column='dt'):
    """
    Attempts to detect the date format from the first few rows of data
    Returns a user-friendly format like DD/MM/YYYY
    """
    # Your existing function code unchanged...
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
    
    # Get a sample date value (first non-null)
    sample_dates = df[date_column].dropna()
    if len(sample_dates) == 0:
        return "DD/MM/YYYY"  # Default if no valid dates
        
    sample = str(sample_dates.iloc[0])
    
    # Simple pattern detection based on common formats
    if re.match(r'\d{4}-\d{2}$', sample):
        return "YYYY-MM"  # Year-month format like "1956-01"
    elif re.match(r'\d{4}-\d{2}-\d{2}', sample):
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
    
    # Generate unique filename for S3
    temp_filename = f"temp_{uuid.uuid4().hex}{file_ext}"
    
    try:
        # Read the uploaded file with appropriate method
        if file_ext == '.csv':
            df = pd.read_csv(file)
        elif file_ext == '.xlsx':
            df = pd.read_excel(file, engine='openpyxl')
        elif file_ext == '.xls':
            df = pd.read_excel(file, engine='xlrd')
        else:
            return "Unsupported file format", 400
        
        # Validate that the file contains data
        if len(df) == 0:
            return "The uploaded file doesn't contain any data", 400
        
        # Display first few rows for debugging
        print(f"[DEBUG] File type: {file_ext}, First few rows:")
        print(df.head())
        
        # Save processed dataframe to S3 as CSV
        csv_temp_filename = f"temp_{uuid.uuid4().hex}.csv"
        if not save_dataframe_to_s3(df, csv_temp_filename):
            return "Error saving file to storage", 500
        
        # Use the CSV version for further processing
        temp_filename = csv_temp_filename
        
    except Exception as e:
        print(f"[ERROR] File processing error: {str(e)}")
        return f"Error processing file: {str(e)}", 400
    
    # Redirect to the config page with the temporary filename
    return redirect(url_for('configure', filename=temp_filename))

@app.route('/configure/<filename>')
def configure(filename):
    # Try to detect date format from the file
    detected_format = "DD/MM/YYYY"  # Default
    
    try:

        # Read sample from S3
        df_sample = read_from_s3(filename)
        if df_sample is not None:
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
    
    try:
        df = read_from_s3(filename)
        if df is None:
            return "Error reading file from storage", 500
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
    
    # Special handling for Year-Month format
    if date_format == "YYYY-MM":
        # For Year-Month format, we need to create a proper datetime
        # Add day component (1st day of month) if missing
        df['dt'] = pd.to_datetime(df['dt'] + '-01', errors='coerce', format=python_date_format + '-%d')
    else:
        # Normal date parsing
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce', format=python_date_format)
    
    print(f"\n[DEBUG] Python date format: {python_date_format}")
    try:
        df['dt'] = pd.to_datetime(df['dt'], format=python_date_format)
    except ValueError as e:
        return f"Could you go back and check your date format and make sure its correct?", 400
    
    # Save the processed file to S3
    processed_filename = f"processed_{filename}"
    if not save_dataframe_to_s3(df, processed_filename):
        return "Error saving processed file", 500
    
    # Delete the original temporary file
    delete_from_s3(filename)
    
    # Now redirect to forecast with the processed file
    return redirect(url_for('forecast', filename=processed_filename, steps=steps))

@app.route('/forecast/<filename>')
def forecast(filename):
    steps = request.args.get('steps', default=10, type=int)
    domain = request.args.get('domain', default=None)
    
    app.logger.info(f"Starting forecast for file: {filename}, steps: {steps}, domain: {domain}")
    
    # Read the processed file from S3
    try:
        app.logger.info(f"Reading file from S3: {filename}")
        df = read_from_s3(filename)
        if df is None:
            app.logger.error(f"Error reading processed file - returned None: {filename}")
            return "Error reading processed file from storage", 500
        app.logger.info(f"Successfully read file, shape: {df.shape}")
    except Exception as e:
        app.logger.error(f"Exception reading file from S3: {str(e)}", exc_info=True)
        return f"Error reading processed file: {str(e)}", 500
    
    try:
        app.logger.info("Cleaning column names")
        df.columns = [str(c).strip().replace("=", "").replace("(", "").replace(")", "") for c in df.columns]
        app.logger.info(f"Columns after cleaning: {df.columns.tolist()}")
        
        app.logger.info("Preparing data for forecasting")
        prepped_data = prepare_data(df.copy())
        app.logger.info(f"Data prepared, length: {len(prepped_data)}")
        
        app.logger.info(f"Selecting forecast model for domain: {domain}")
        forecast_function, model_name = select_forecasting_model(prepped_data, domain, return_name=True)
        app.logger.info(f"Selected model: {model_name}")
        
        app.logger.info(f"Generating forecast for {steps} steps")
        forecast_df = forecast_function(prepped_data, forecast_steps=steps)
        app.logger.info(f"Forecast generated, shape: {forecast_df.shape}")
        
        print(f"\n[DEBUG] Selected model: {model_name}")
        print(f"[DEBUG] Data length: {len(prepped_data)}")
        print(f"[DEBUG] Domain: {domain if domain else 'Not specified'}")
        print(f"[DEBUG] Forecast steps: {steps}\n")

        # Create an interactive Plot
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

        # Save forecast to S3
        forecast_filename = f"forecast_{filename}"
        if not save_dataframe_to_s3(forecast_df, forecast_filename):
            return "Error saving forecast file", 500
        
        expected_accuracy = round(calculate_cv_accuracy(
            prepped_data,
            forecast_function,
            forecast_horizon=steps,
            num_folds=2,
            stride=1
        )['mape'], 3)

        baseline_accuracy = round(calculate_cv_accuracy(
            prepped_data,
            baseline_forecast,
            forecast_horizon=steps,
            num_folds=2,
            stride=1
        )['mape'], 3)
        
        # Clean up the processed file after rendering the template
        @after_this_request
        def cleanup(response):
            try:
                # Delete the processed file from S3
                delete_from_s3(filename)
                print(f"[CLEANUP] Removed processed file: {filename}")
                
                # Delete the original file (removing 'processed_' prefix) if it exists
                original_filename = filename.replace('processed_', '')
                if original_filename != filename:
                    delete_from_s3(original_filename)
                    print(f"[CLEANUP] Removed original file: {original_filename}")
            except Exception as e:
                print(f"[ERROR] Failed to clean up files: {e}")
            return response
    except Exception as e:
        app.logger.error(f"Exception during forecasting: {str(e)}", exc_info=True)
        return f"Error during forecasting: {str(e)}", 500

    return render_template('result.html', plot_html=plot_html, forecast_file=forecast_filename, expected_accuracy=expected_accuracy, baseline_accuracy=baseline_accuracy)

@app.route('/download/<path:forecast_file>')
def download_file(forecast_file):
    try:
        # Generate a pre-signed URL for the forecast file
        presigned_url = generate_presigned_url(forecast_file)
        if not presigned_url:
            return "Error generating download link", 500
        
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
