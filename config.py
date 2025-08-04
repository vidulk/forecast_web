"""
Configuration settings for the forecast web application.
"""
import os

# Flask configuration
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max upload size

# S3 configuration
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# Granularity mapping
GRANULARITY_MAP = {
    'hourly': 'H',
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'M'
}

# Seasonality mapping
SEASONALITY_MAP = {
    'hourly': 24,
    'daily': 7,
    'weekly': 52,
    'monthly': 12
}

# Date format mapping
DATE_FORMAT_MAP = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "DD-MM-YYYY": "%d-%m-%Y",
    "MM-DD-YYYY": "%m-%d-%Y",
    "YYYY/MM/DD": "%Y/%m/%d",
    "DD/MM/YYYY": "%d/%m/%Y",
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD-MM-YY": "%d-%m-%y",
    "MM-DD-YY": "%m-%d-%y",
    "DD/MM/YY": "%d/%m/%y",
    "MM/DD/YY": "%m/%d/%y",
    "YYYY-MM": "%Y-%m",
    "DD-MON-YYYY": "%d-%b-%Y",
    "MON DD, YYYY": "%b %d, %Y",
}
