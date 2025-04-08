import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from ann_model import ANN
import os
import warnings

# Redirect warnings to stderr
warnings.filterwarnings("default", module="yfinance")
warnings.simplefilter("always")

def download_data(company_name, start_date, end_date):
    """
    Downloads stock data using yfinance.
    """
    try:
        # Suppress progress messages and explicitly set auto_adjust
        data = yf.download(company_name, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError("No stock data found for the given ticker and date range")
        return data
    except Exception as e:
        raise ValueError(f"Error downloading stock data: {str(e)}")

def prepare_data(stock_data, time_step=100):
    """
    Prepares the data for training by creating input-output pairs.
    """
    close_prices = stock_data['Close'].values  # Extract the 'Close' prices
    if len(close_prices) < time_step:
        raise ValueError("Not enough data to prepare input-output pairs")

    X, y = [], []
    for i in range(time_step, len(close_prices)):
        X.append(close_prices[i - time_step:i])  # Input: Previous `time_step` prices
        y.append(close_prices[i])  # Output: The next price

    return np.array(X), np.array(y)

def normalize_data(data):
    """
    Normalizes the data to the range [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if min_val == max_val:
        raise ValueError("Normalization failed: min and max values are the same")
    return (data - min_val) / (max_val - min_val), min_val, max_val

def calculate_accuracy(actual, predicted):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) as accuracy.
    """
    return 100 - np.mean(np.abs((actual - predicted) / actual)) * 100

def train_ann(ann, X_normalized, y_normalized, epochs=1000, learning_rate=0.01, patience=50):
    """
    Trains the ANN with early stopping and learning rate adjustment.
    """
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        ann.forward(X_normalized)
        ann.backward(X_normalized, y_normalized, learning_rate)
        loss = np.mean(np.square(y_normalized - ann.output))

        if epoch % 100 == 0:
            sys.stderr.write(f'Epoch {epoch}, Loss: {loss:.4f}\n')

        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                sys.stderr.write(f"Early stopping at epoch {epoch}, Best Loss: {best_loss:.4f}\n")
                break

        # Adjust learning rate
        if epoch > 0 and epoch % 200 == 0:
            learning_rate *= 0.9  # Reduce learning rate by 10%
            sys.stderr.write(f"Learning rate adjusted to {learning_rate:.6f}\n")

def main():
    try:
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Missing required arguments: ticker and prediction_date"}))
            sys.exit(1)

        ticker = sys.argv[1]
        prediction_date = sys.argv[2]

        today = datetime.now()
        yesterday = today - timedelta(days=1)
        yesterdays_date = yesterday.strftime('%Y-%m-%d')

        # Download stock data
        stock_data = download_data(ticker, '2020-01-01', yesterdays_date)

        if stock_data.empty:
            print(json.dumps({"error": "No stock data found for the given ticker and date range"}))
            sys.exit(1)

        # Prepare data
        time_step = 100  # Experiment with different values (e.g., 50, 150)
        hidden_size = 64
        output_size = 1
        X, y = prepare_data(stock_data, time_step)
        X_normalized, X_min, X_max = normalize_data(X)
        y_normalized, y_min, y_max = normalize_data(y)
        X_normalized = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1])
        input_size = X_normalized.shape[1]

        # Initialize and train ANN
        ann = ANN(input_size, hidden_size, output_size)
        train_ann(ann, X_normalized, y_normalized, epochs=1000, learning_rate=0.01, patience=50)

        # Make predictions
        prediction_data = stock_data['Close'][-time_step:].values  # Use the last `time_step` prices
        prediction_data = prediction_data.reshape(1, -1)
        prediction_data_normalized = (prediction_data - X_min) / (X_max - X_min)
        predicted_price_normalized = ann.forward(prediction_data_normalized)
        predicted_price = predicted_price_normalized * (y_max - y_min) + y_min

        # Calculate accuracy
        actual_prices = stock_data['Close'].values
        predicted_prices = ann.forward(X_normalized) * (y_max - y_min) + y_min
        accuracy = calculate_accuracy(actual_prices[time_step:], predicted_prices)

        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices[time_step:], label='Actual Price', color='blue')
        plt.plot(predicted_prices, label='Predicted Price', color='red')
        plt.title(f'{ticker} Stock Analysis')
        plt.xlabel('Days')
        plt.ylabel('Price')
        graph_dir = "graphs"
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        graph_path = f"{graph_dir}/{ticker}_prediction.png"
        plt.savefig(graph_path)

        # Output result as JSON
        predicted_value = round(float(predicted_price[0][0]), 2)
        response = {
            "predicted_price": predicted_value,
            "graph_path": graph_path,
            "accuracy": round(accuracy, 2)
        }
        print(json.dumps(response))
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()