"""
Route handlers for the forecast web application.
"""
import os
import uuid
import pandas as pd
from flask import (
    request, render_template, redirect, url_for, jsonify, 
    send_from_directory, after_this_request, Response, current_app
)
from utils.timing import time_function
from utils.s3_client import (
    read_from_s3, save_dataframe_to_s3, delete_from_s3, get_s3_client
)
from utils.data_utils import (
    detect_columns, detect_date_format, detect_granularity, convert_date_format
)
from services.forecast_service import process_forecast_request
from config import S3_BUCKET_NAME


def register_routes(app):
    """Register all routes with the Flask app"""
    
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
        all_columns = []
        guessed_covariates_str = ""
        
        try:
            # Read sample from S3
            df_sample = read_from_s3(filename)
            if df_sample is not None:
                all_columns = df_sample.columns.tolist()
                date_col, _, guessed_covariates = detect_columns(df_sample)
                
                detected_format = detect_date_format(df_sample, date_column=date_col)
                detected_granularity = detect_granularity(df_sample.copy())
                guessed_covariates_str = ",".join(guessed_covariates)
                
                print(f"[DEBUG] Detected date format: {detected_format}")
                print(f"[DEBUG] Detected granularity: {detected_granularity}")
                print(f"[DEBUG] Guessed covariates: {guessed_covariates_str}")

        except Exception as e:
            print(f"[WARNING] Could not detect date format or granularity: {e}")
        
        return render_template('configure.html', 
                               filename=filename, 
                               all_columns=all_columns,
                               detected_format=detected_format,
                               detected_granularity=detected_granularity,
                               guessed_covariates=guessed_covariates_str)

    @app.route('/process', methods=['POST'])
    @time_function("PROCESS_ROUTE")
    def process():
        filename = request.form.get('filename')
        date_column = request.form.get('date_column', 'ds')
        date_format = request.form.get('date_format', 'DD/MM/YYYY')
        value_column = request.form.get('value_column', 'value')
        steps = request.form.get('steps', type=int, default=7)
        granularity = request.form.get('granularity', 'daily')
        use_covariates = 'use_covariates' in request.form
        covariate_columns_str = request.form.get('covariate_columns', '')
        
        if not steps or steps <= 0:
            return "Invalid timesteps value", 400
        
        try:
            df = read_from_s3(filename)
            if df is None:
                return "Error reading file from storage", 500
                
            # Rename columns to the expected formats
            if date_column != 'ds':
                df = df.rename(columns={date_column: 'ds'})
            if value_column != 'y':
                df = df.rename(columns={value_column: 'y'})
            
            # Handle covariates
            if use_covariates and covariate_columns_str:
                covariate_columns = [c.strip() for c in covariate_columns_str.split(',')]
                # Prefix covariate columns to avoid name clashes, e.g., with 'y' or 'ds'
                rename_dict = {col: f"x_{col}" for col in covariate_columns if col in df.columns}
                df = df.rename(columns=rename_dict)
            
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
                current_app.logger.error(f"Date conversion failed with format: {date_format}. Resulting 'ds' column has nulls.")
                error_message = "The date format you selected doesn't seem to match your data. Please check the format and try again."
                return render_template('error.html', message=error_message, filename=filename), 400
            
            processed_filename = f"processed_{filename}"
            if not save_dataframe_to_s3(df, processed_filename):
                return "Error saving processed file", 500
            
            # Delete the original temporary file
            delete_from_s3(filename)
            
            # Preserve all original parameters for the next step
            query_params = request.form.to_dict()
            query_params.pop('filename', None) # filename is in the URL path
            
            return redirect(url_for('forecast_loading', filename=processed_filename, **query_params))
            
        except Exception as e:
            return f"Error processing file: {str(e)}", 400

    @app.route('/forecast_loading/<filename>')
    def forecast_loading(filename):
        """Renders a loading page that will trigger the forecast generation."""
        return render_template('loading.html', filename=filename)

    @app.route('/api/forecast/<filename>')
    @time_function("FORECAST_API")
    def forecast_api(filename):
        try:
            steps = request.args.get('steps', default=10, type=int)
            granularity_str = request.args.get('granularity', default='daily')
            domain = request.args.get('domain', default=None)
            use_covariates = request.args.get('use_covariates', 'false').lower() == 'true'

            forecast_df, expected_accuracy, model_name, plot_html, baseline_accuracy = process_forecast_request(
                filename, steps, granularity_str, domain, use_covariates
            )
            
            # Save forecast to S3
            forecast_filename = f"forecast_{filename}"
            if not save_dataframe_to_s3(forecast_df, forecast_filename):
                return jsonify({'error': "Error saving forecast file"}), 500

            # Save plot to S3 to be retrieved by the result page
            plot_filename = f"plot_{filename}.html"
            s3 = get_s3_client()
            s3.put_object(Bucket=S3_BUCKET_NAME, Key=plot_filename, Body=plot_html.encode('utf-8'))

            result_url = url_for('forecast_result', 
                                 filename=filename, 
                                 forecast_file=forecast_filename,
                                 plot_file=plot_filename,
                                 expected_accuracy=expected_accuracy,
                                 baseline_accuracy=baseline_accuracy)

            return jsonify({'result_url': result_url})
            
        except Exception as e:
            current_app.logger.error(f"Exception during forecasting API call: {str(e)}", exc_info=True)
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
            current_app.logger.error(f"Error rendering forecast result: {str(e)}", exc_info=True)
            return "Error displaying forecast result.", 500

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

    @app.route('/sample/<path:filename>')
    def download_sample(filename):
        """Route to serve the sample CSV file for download."""
        sample_directory = os.path.join(current_app.root_path, 'sample')
        return send_from_directory(sample_directory, filename, as_attachment=True)

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle all unhandled exceptions and log them properly"""
        current_app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}", 500
