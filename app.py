from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Universe of assets to compare
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', '^NSEI', '^DJI', 'TSLA']

# Serve frontend
@app.route('/')
def home():
    return render_template('index.html')

# Check if interval is intraday
def is_intraday(interval):
    intraday_intervals = ['1m','3m','5m','10m','15m','20m','30m','45m','60m','180m','360m']
    return interval in intraday_intervals

# Compute similarity using DTW
def compute_similarity(target_df, compare_df):
    distance, _ = fastdtw(target_df['Close'], compare_df['Close'], dist=euclidean)
    max_len = max(len(target_df), len(compare_df))
    similarity = max(0, 100 * (1 - distance / max_len))
    return similarity

# Compare endpoint
@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    asset = data.get('asset', '').upper()
    interval = data.get('timeframe', '1d')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    # Validate input
    if not asset or not start_date or not end_date:
        return jsonify({'error': 'Asset, start date, and end date are required'}), 400

    # Convert dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        return jsonify({'error': 'Invalid date format'}), 400

    # For intraday, restrict to last 7 days
    if is_intraday(interval):
        seven_days_ago = datetime.today() - timedelta(days=7)
        if start_dt < seven_days_ago:
            start_dt = seven_days_ago
        if end_dt > datetime.today():
            end_dt = datetime.today()

    # Fetch target asset data
    try:
        target_df = yf.download(asset, start=start_dt, end=end_dt, interval=interval, progress=False)
        if target_df.empty:
            return jsonify({'error': f'No data for {asset} in this range/interval'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    results = []
    for sym in UNIVERSE:
        try:
            compare_df = yf.download(sym, start=start_dt, end=end_dt, interval=interval, progress=False)
            if compare_df.empty:
                continue
            sim = compute_similarity(target_df, compare_df)
            if sim > 80:  # Only show similarity > 80%
                results.append({'asset': sym, 'similarity': sim})
        except:
            continue

    # Sort by similarity descending
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify({'results': results})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)