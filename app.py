import os
import pandas as pd
import plotly.graph_objs as go
from plotly.io import to_html
from flask import Flask, render_template, request, redirect, url_for, send_file
from forecasting import select_forecasting_model, baseline_forecast, calculate_cv_accuracy, prepare_data
from datetime import timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'forecasts'

# Ensure forecasts folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    steps = request.form.get('steps', type=int)
    if not steps or steps <= 0:
        return "Invalid timesteps value", 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Pass the number of steps to the forecast route
        return redirect(url_for('forecast', filename=file.filename, steps=steps))

@app.route('/forecast/<filename>')
def forecast(filename):
    steps = request.args.get('steps', default=10, type=int)  # Default to 10 if not specified
    domain = request.args.get('domain', default=None)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df['dt'] = pd.to_datetime(df['dt'], format='%d/%m/%y')

    prepped_data = prepare_data(df.copy())
    forecast_function, model_name = select_forecasting_model(prepped_data, domain, return_name=True)
    forecast_df = forecast_function(prepped_data, forecast_steps=steps)
    
    print(f"\n[DEBUG] Selected model: {model_name}")
    print(f"[DEBUG] Data length: {len(prepped_data)}")
    print(f"[DEBUG] Domain: {domain if domain else 'Not specified'}")
    print(f"[DEBUG] Forecast steps: {steps}\n")

    # Create an interactive Plo
    fig = go.Figure()
    df['dt'] = df['dt'].dt.strftime('%Y-%m-%d')
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
