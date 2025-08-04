"""
Forecast Web Application - Main Flask Application

A modular web application for time series forecasting with S3 storage.
"""
from flask import Flask
from config import MAX_CONTENT_LENGTH
from routes.main_routes import register_routes
from utils.s3_client import cleanup_old_s3_files


def create_app():
    """Application factory pattern for creating Flask app"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    # Register all routes
    register_routes(app)
    
    return app


# Create the Flask app
app = create_app()


if __name__ == '__main__':
    cleanup_old_s3_files()  # Clean up old files on startup
    app.run(debug=True)
