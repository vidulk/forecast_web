# Forecast Web Application - Refactored Architecture

This is a modular, scalable Flask web application for time series forecasting with AWS S3 storage integration.

## Project Structure

```
forecast_web/
├── app.py                      # Main Flask application (entry point)
├── config.py                   # Configuration settings
├── forecasting.py              # Core forecasting models and algorithms
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python runtime version
├── Procfile                    # Deployment configuration
├── 
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── timing.py               # Performance timing decorators
│   ├── s3_client.py            # AWS S3 operations
│   └── data_utils.py           # Data processing utilities
├── 
├── services/                   # Business logic services
│   ├── __init__.py
│   └── forecast_service.py     # Forecasting business logic
├── 
├── routes/                     # HTTP route handlers
│   ├── __init__.py
│   └── main_routes.py          # Main application routes
├── 
├── templates/                  # HTML templates
│   ├── upload.html
│   ├── configure.html
│   ├── loading.html
│   ├── result.html
│   └── error.html
├── 
├── static/                     # Static files (CSS, images, etc.)
│   ├── style.css
│   ├── logo.png
│   └── sample.csv
├── 
└── [generated folders]
    ├── forecasts/              # Generated forecast files
    ├── temp_files/             # Temporary processing files
    └── sample/                 # Sample data files
```

## Architecture Overview

### Modular Design

The application has been refactored into a clean, modular architecture:

1. **`app.py`** - Main entry point using Flask application factory pattern
2. **`config.py`** - Centralized configuration management
3. **`utils/`** - Reusable utility functions
4. **`services/`** - Business logic separated from routes
5. **`routes/`** - HTTP request handling

### Key Modules

#### Configuration (`config.py`)
- Environment variables management
- AWS S3 settings
- Granularity and seasonality mappings
- Date format mappings

#### Utils Package
- **`timing.py`** - Performance monitoring decorators
- **`s3_client.py`** - AWS S3 operations (upload, download, delete, cleanup)
- **`data_utils.py`** - Data processing, format detection, and preparation

#### Services Package
- **`forecast_service.py`** - Core forecasting business logic, model selection, and prediction

#### Routes Package
- **`main_routes.py`** - All HTTP route handlers with proper separation of concerns

## Benefits of This Architecture

### 1. **Modularity**
- Each module has a single responsibility
- Easy to test individual components
- Clear separation between data processing, business logic, and web handling

### 2. **Scalability**
- Easy to add new features without modifying existing code
- Services can be extracted to microservices if needed
- Clear interfaces between components

### 3. **Maintainability**
- Reduced code duplication
- Clear organization makes it easy to find and modify code
- Configuration centralized in one place

### 4. **Readability**
- Main `app.py` is now concise and focused
- Each file has a clear purpose
- Function names and module organization are self-documenting

## Usage

The application functionality remains exactly the same:

1. **Upload** time series data (CSV/Excel)
2. **Configure** date formats, columns, and forecasting parameters
3. **Process** data and generate forecasts
4. **View** results with interactive plots
5. **Download** forecast data

## Development

To run the application:

```bash
cd forecast_web
python app.py
```

The refactored structure makes it easy to:
- Add new route handlers in `routes/`
- Add new utility functions in `utils/`
- Extend forecasting capabilities in `services/`
- Modify configuration in `config.py`

## Environment Variables

The application requires these environment variables for S3 integration:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET_NAME`
