import os
import pandas as pd
import plotly.graph_objs as go
from plotly.io import to_html
from flask import Flask, render_template, request, redirect, url_for, send_file
from forecasting import forecast_with_theta
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
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Placeholder for your forecasting logic
    try:
        forecast_df = forecast_with_theta(df, forecast_steps=steps)
    except Exception as e:
        return f"An error occurred during forecasting: {e}", 500


    # Create an interactive Plo
    fig = go.Figure()
    df['dt'] = df['dt'].dt.strftime('%Y-%m-%d')
    fig.add_trace(go.Scatter(x=df['dt'], y=df['value'], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=forecast_df['dt'], 
                             y=forecast_df['value'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title='Forecast vs Original', xaxis_title='Index', yaxis_title='Value')
    
    # Add config to hide the mode bar
    config = {
        'displayModeBar': False,  # Hides the toolbar completely
    }
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False, config=config)

    forecast_path = os.path.join(app.config['UPLOAD_FOLDER'], f'forecast_{filename}')
    forecast_df.to_csv(forecast_path, index=False)

    return render_template('result.html', plot_html=plot_html, forecast_file=forecast_path)

@app.route('/download/<path:forecast_file>')
def download_file(forecast_file):
    return send_file(forecast_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
