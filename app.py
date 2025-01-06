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
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('forecast', filename=file.filename))

@app.route('/forecast/<filename>')
def forecast(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Placeholder for your forecasting logic
    try:
        forecast_df = forecast_with_theta(df, forecast_steps=10)
    except Exception as e:
        return f"An error occurred during forecasting: {e}", 500

    # Create an interactive Plo
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=forecast_df.index, 
                             y=forecast_df['value'], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title='Forecast vs Original', xaxis_title='Index', yaxis_title='Value')

    # Generate HTML for Plotly plot
    plot_html = to_html(fig, full_html=False)

    forecast_path = os.path.join(app.config['UPLOAD_FOLDER'], f'forecast_{filename}')
    forecast_df.to_csv(forecast_path, index=False)

    return render_template('result.html', plot_html=plot_html, forecast_file=forecast_path)

@app.route('/download/<path:forecast_file>')
def download_file(forecast_file):
    return send_file(forecast_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
