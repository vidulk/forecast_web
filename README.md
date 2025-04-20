📈 Forecast For Everyone

Forecast For Everyone is a simple but powerful web app that lets anyone — upload their own data and receive intelligent, benchmarked forecasts, backed by real-world experience and smart automation.

📁 Upload your CSV → 🧠 Answer a few quick questions → 📊 Get forecasts, graphs, and downloads

🔍 What It Does

📤 Upload CSV/XLSX: Users upload their own time series data.
🤖 Smart Forecasting Engine: Automatically selects the most suitable forecasting model based on:
Data length
Granularity
Seasonality
Stationarity
Complexity / intermittency
📄 Guided Questions: Prompts the user for helpful context (e.g., date column, frequency, units).
📈 Outputs:
Forecast plot
Key metrics (like RMSE)
Downloadable forecast (CSV)
Comparison with a baseline (e.g. last 4-week average)
🧠 How Forecasting Works

Forecast selection is rule-based, designed using practical forecasting experience. Depending on the data, it may choose:

LightGBM (for structured, high-volume data)
Time series models (e.g., exponential smoothing, moving averages)
Seasonal naive or average-based models
Prophet, ARIMA, etc
The goal is intelligent automation: simple interface, powerful backend.

🔧 Tech Stack

Layer	Tool
Backend	Flask
Forecasting Engine	Multiple models (selected by rules)
Frontend	HTML + Jinja2 templates, Bootstrap
Visualization	Plotly
Hosting	Heroku
Language	Python-only 💯

🛣️ Roadmap

Coming Soon:

✅ Support for future & past covariates (e.g. price, inventory, promotions)
✅ Automatic holiday / weather features
✅ Better validation visualizations & diagnostic plots
✅ Save user sessions + accounts

🚀 Getting Started

Clone the repo
git clone https://github.com/vidulk/forecast_web.git
cd forecast_web
Install dependencies
pip install -r requirements.txt
Run the app
python app.py
Open your browser at: http://localhost:5000
🌐 Live Demo

⚙️ Hosted on Heroku – coming soon!
📬 Contact

Built with ❤️ by a data scientist making forecasting accessible, practical, and no-code.

GitHub: @vidulk
LinkedIn: vidulk7
Email: vidulkhanna@gmail.com
