import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def fetch_data(symbol, days, feature='Close', interval='1d'):
    print("Downloading data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
    return data[[feature]]

def add_lag_features(data, num_lags):
    df = pd.DataFrame(data)
    for lag in range(1, num_lags + 1):
        df[f'lag_{lag}'] = df.iloc[:, 0].shift(lag)
    df.dropna(inplace=True)
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, num_features, num_layers, neurons_per_layer, learning_rate):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(neurons_per_layer[i], return_sequences=(i < num_layers - 1), input_shape=(input_shape, num_features)))
        else:
            model.add(LSTM(neurons_per_layer[i], return_sequences=(i < num_layers - 1)))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model

def simulate_investment(predictions, actual_prices, initial_investment, reinvest_percentage, leverage_factor, daily_interest_rate, stop_loss_percentage):
    cash = initial_investment
    holdings = 0
    buy_price = 0  # Track the price at which stocks were last bought
    trading_history = []  # To store detailed trading history

    for day, (predicted_price, actual_price) in enumerate(zip(predictions, actual_prices)):
        if holdings > 0:
            # Check if the current price triggers the stop-loss
            if (actual_price - buy_price) / buy_price < -stop_loss_percentage:
                cash = holdings * actual_price
                holdings = 0
                trading_history.append({'Day': day, 'Actual Price': actual_price, 'Predicted Price': predicted_price, 'Action': 'Stop-Loss Sell', 'Holdings': holdings, 'Cash': cash})
                continue  # Skip further trading checks for this cycle

        if predicted_price > actual_price and cash > 0:
            # Buy decision
            if leverage_factor > 1:
                cash *= leverage_factor
            holdings = cash / actual_price
            cash = 0
            buy_price = actual_price  # Update buy price on purchase
            trading_history.append({'Day': day, 'Actual Price': actual_price, 'Predicted Price': predicted_price, 'Action': 'Buy', 'Holdings': holdings, 'Cash': cash})
        elif predicted_price < actual_price and holdings > 0:
            # Sell decision
            cash = holdings * actual_price
            holdings = 0
            profit = cash - initial_investment
            reinvest_amount = profit * reinvest_percentage
            cash = initial_investment + reinvest_amount
            cash *= (1 + daily_interest_rate)  # Apply daily interest rate to the remaining cash
            trading_history.append({'Day': day, 'Actual Price': actual_price, 'Predicted Price': predicted_price, 'Action': 'Sell', 'Holdings': holdings, 'Cash': cash})

    if holdings > 0:
        cash = holdings * actual_prices[-1]
        trading_history.append({'Day': len(predictions), 'Actual Price': actual_prices[-1], 'Predicted Price': predictions[-1], 'Action': 'Final Sell', 'Holdings': holdings, 'Cash': cash})

    trading_results_df = pd.DataFrame(trading_history)
    return cash, trading_results_df

def main():
    symbol = input("Enter the stock ticker (e.g., 'AAPL'): ")
    days = int(input("Enter the total number of days of data to analyze (e.g., 1000): "))
    feature = input("Choose the feature for prediction (Close, Open, Volume): ")
    interval = input("Enter the data interval (e.g., '1d' for daily): ")
    window_size = int(input("Enter the number of days each data point should be trained on (e.g., 60): "))
    num_lags = int(input("Enter the number of lag features to include (e.g., 2): "))
    num_layers = int(input("Enter the number of LSTM layers (e.g., 2): "))
    neurons_per_layer = [int(input(f"Enter the number of neurons for layer {i + 1}: ")) for i in range(num_layers)]
    epochs = int(input("Enter the number of epochs for training the model (e.g., 50): "))
    learning_rate = float(input("Enter the learning rate (e.g., 0.001): "))
    initial_investment = float(input("Enter the amount of money you want to simulate investing (e.g., 1000): "))
    train_split = float(input("Enter the train/test split ratio (e.g., 0.8): "))
    reinvest_percentage = float(input("Enter the percentage of profits to reinvest (default 1 for 100%): ") or 1)
    leverage_factor = float(input("Enter the leverage factor (default 1, e.g., 1.5 for leverage): ") or 1)
    annual_interest_rate = float(input("Enter the annual interest rate for cash reserves (default 0, e.g., 0.02 for 2%): ") or 0)
    daily_interest_rate = annual_interest_rate / 365
    stop_loss_percentage = float(input("Enter the stop loss percentage (e.g., 0.1 for 10%): "))

    data = fetch_data(symbol, days, feature, interval)
    data_with_lags = add_lag_features(data, num_lags)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_with_lags)
    X, y = create_sequences(scaled_data, window_size)

    split_point = int(len(X) * train_split)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    model = build_lstm_model(window_size, X.shape[2], num_layers, neurons_per_layer, learning_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    predictions_expanded = np.zeros((predictions.shape[0], scaled_data.shape[1]))
    predictions_expanded[:, 0] = predictions.flatten()  # Assuming predictions are for the first feature
    predicted_prices = scaler.inverse_transform(predictions_expanded)[:, 0]
    actual_prices = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1))]))[:, 0]

    final_cash, trading_results_df = simulate_investment(predicted_prices, actual_prices, initial_investment, reinvest_percentage, leverage_factor, daily_interest_rate, stop_loss_percentage)

    # Save trading results to CSV
    trading_results_df.to_csv('trading_results.csv', index=False)

    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices', linestyle='--')
    plt.title(f'{symbol} Stock Price Prediction on Test Data')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"Final cash after trading on test data: ${final_cash:.2f}")

    print(f"Final cash after trading on test data: ${final_cash:.2f}")

if __name__ == '__main__':
    main()
