# File: stock_prediction.py
#   Base-Project Authors: Cheong Koo and Bao Vo
#   Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

#   Original code modified from:
#   Title: Predicting Stock Prices with Python
#   Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
#   By: NeuralNine

#   Current-Project Author: 102155889
#   For class: COS30018 - Intelligent Systems

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import nd_grid

import pandas as pd
import os
import json
import datetime as dt
import statsmodels.api as sm
import pmdarima as pm

import tensorflow as tf
import yfinance as yf
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def save_settings(settings, name_or_path):
    """
    Saves the given settings dictionary to a JSON file with the given name or path.

    Params:
        settings (dict): The dictionary of settings to save.
        name_or_path (str): The name or path of the file to save the settings to.
    """
    if not name_or_path.endswith('.json'):
        name_or_path += '.json'
    with open(name_or_path, 'w', encoding='utf-8') as file:
        # Convert each setting to a string before serializing for JSON
        settings_str = {key: str(value) for key, value in settings.items()}
        json.dump(settings_str, file, indent=4)

# ------------------------------------------------------------------------------
# Settings and Parameters
# ------------------------------------------------------------------------------
TEST_NAME = "Final_Test"
RUN_MODEL = True

PLOT_LOSS = True
SAVE_LOSS_PLOT = True
DISPLAY_PLOTS = False

COMPARE_GLOBAL_PARAMS_LIST = ['PRICE_VALUE', 'SCALE_FEATURES', 'RESAMPLE_DATASET', 'RESAMPLE_FREQ', 'PAD_MISSING_DAYS', 'TEST_SIZE']

COMPARISON_PARAMS_DICT = {
    "LSTM": ['CELL_NAME','EPOCHS','BATCH_SIZE','DROPOUT','LOSS','OPTIMIZER', 'NUM_LAYERS','UNITS', 'PREDICT_INCR_PAST','PREDICT_INCR_AHEAD', 'FEATURE_COLUMNS'],
    "ARIMA" : ['FORECAST_METHOD', 'ARIMA_OOS_FORECASTS', 'arima_order', ],
    "SARIMA" : ['FORECAST_METHOD', 'ARIMA_OOS_FORECASTS', 'sarima_order','sarima_seasonal_order', 'SEASONAL_PERIOD']}

COMPARE_LOSS = True
COMPARE_PREDICTIONS = True
COMPARISON_PREDICTION_MODELS = ["LSTM", "ARIMA", "SARIMA", "ENSEMBLE"]
COMPARISON_PREDICTION_COLUMN = {
    "Close" : "Price Value ($)", 
    "price_error" : "Price Error ($)", 
    "price_error_percent" : "Price Error (%)"}

PLOT_ENSEMBLE = True
PLOT_INDIVIDUAL_RUN = True
COMPARE_RUNS = True
CANDLE_VIS_WINDOW = 10
BOX_VIS_WINDOW = 90

CELL = LSTM  # the type of cell to use for the DL network | LSTM, or GRU (untested)             | Default: LSTM

SETTINGS = {
    # Get variable values for the plot and filename,
    'TEST_NAME' : TEST_NAME,
    'DATA_SOURCE' : "yahoo",
    'COMPANY' : "TSLA", # TSLA, AAPL, etc.

    # Prediction parameters
    'PREDICT_INCR_PAST' : 52, # number of past increments to base the prediction from   | Default: 60
    'PREDICT_INCR_AHEAD' : 1, # number of increments ahead to base our prediction       | Default: 1

    # 'real', 'speculative' | Only takes effect when ARIMA_OOS_FORECASTS is set to -1 (auto)
    'FORECAST_METHOD' : 'speculative', # Forecasting ARIMA/SARIMA models ('real', 'speculative')       | Default: 'real'
    # real: Trained on test data up until PREDICT_INCR_PAST before the current prediction,
    #               forecasting PREDICT_INCR_AHEAD ahead using real historical data for each forecast.
    # speculative: Not trained on test data, forecasting PREDICT_INCR_AHEAD ahead,
    #               using prior predictions, as history for each forecast.

    # Data parameters
    'TRAIN_START' : '2015-01-01',
    'TRAIN_END' : '2020-01-01',
    'RESAMPLE_DATASET' : True,      # whether to resample the data to be periodic               | Default: True
    'RESAMPLE_FREQ' : 'W',          # the frequency to resample the data ('D', 'W', 'M', etc.)  | Default: 'W'
    'PAD_MISSING_DAYS' : True,      # Pad the missing dates with the last available value
    'TEST_SIZE' : 0.2,              # ratio for test data to training data                      | Default: 0.2
    'SHUFFLE_DATASET' : False,      # whether to shuffle the data                               | Default: True
    'SPLIT_BY_DATE' : True,         # whether to split the data by date                         | Default: True
    'FEATURE_COLUMNS' : ['Open','High', 'Low', 'Close', 'Adj Close', 'Volume'],
    'PRICE_VALUE' : "Close",        # the column to use as the target value                     | Default: "Close"
    'LOCAL_SAVE' : True,            # whether to save the training data to local storage        | Default: True
    'LOCAL_LOAD' : False,           # whether to load the training data from local storage      | Default: True
    'LOCAL_PATH' : './data',        # the path to the local storage                             | Default: './data'
    'SCALE_FEATURES' : True,        #  whether to scale/normalize the training data             | Default: True

    # Model parameters
    'CELL_NAME' : None,             # Here as a stub for the cell name, set later in the script | Don't Modify!
    'EPOCHS' : 25,                  # number of epochs (repetitions of training) to train for.  | Default: 25
    'BATCH_SIZE' : 32,              # number of samples per update.                             | Default: 32
    'LOSS' : "mean_absolute_error", # loss function (mean_absolute_error, mean_squared_error).  | Default: 
    'OPTIMIZER' : 'adam',           # which optimizer function to use (adam, rmsprop, sgd).     | Default: "adam"
    'NUM_LAYERS' : 2,               # number of layers in the DL network.                       | Default: 2
    'DROPOUT' : 0.2,                # defines fraction of deactivated neurons, eg; 0.2 :: 20%.  | Default: 0.2
    'UNITS' : 100,                  # the number of units/nodes in each layer.                  | Default: 50

    # Set which models to run,
    'USE_LSTM' : True,              # whether to use the LSTM model.                            | Default: True
    'USE_ARIMA'  : True,            # whether to use the ARIMA model.                           | Default: True
    'USE_SARIMA' : True,            # whether to use the SARIMA model.                          | Default: True
    'USE_ENSEMBLE' : True,          # whether to use the ensemble model (combination).          | Default: True
    'ENSEMBLE_MODEL' : 'SARIMA',    # Which model to include in the ensemble (ARIMA, SARIMA).   | Default: 'SARIMA'

    # ARIMA parameters
    'LAGS_VAL' : 52,                # essentially the observation window for ACF and PACF plots | Default: 52
    'ARIMA_P' : 1,                  # lag order
    'ARIMA_D' : 1,                  # degree of differencing
    'ARIMA_Q' : 0,                  # order of moving average
    'ARIMA_OOS_FORECASTS': -1,      # Number of out-of-sample forecasts to make | Default: -1
    # A value of 0 means no blind forecasts are made, i.e. the model is trained and tested on the same data.
    # A value of -1 means that the model is trained on all data except for the end period, determined by TEST_SIZE.
    # A value above 0 (i.e. 2) will forecast X increments ahead.

    # SARIMA Gridsearch parameters
    'SARIMA_SEASONAL' : True,
    'SEASONAL_PERIOD' : 52,         # Seasonal period for the SARIMA model
    'SARIMA_d' : None,              # degree of differencing (non-seasonal)                     | Default: None
    'SARIMA_D' : None,              # degree of seasonal differencing                           | Default: None
    'SARIMA_START_p' : 0,           # starting value of p (lag order)                           | Default: 2
    'SARIMA_START_q' : 0,           # starting value of q (order of moving average)             | Default: 2
    'SARIMA_START_P' : 0,           # starting value of P (lag order - seasonal)                | Default: 1
    'SARIMA_START_Q' : 0,           # starting value of Q (order of moving average - seasonal)  | Default: 1
    'SARIMA_MAX_p' : 2,             # maximum value of p (lag order)                            | Default: 5
    'SARIMA_MAX_q' : 2,             # maximum value of q (order of moving average)              | Default: 5
    'SARIMA_MAX_d' : 2,             # maximum value of d (degree of differencing)               | Default: 2
    'SARIMA_MAX_D' : 2,             # maximum value of D (degree of seasonal differencing)      | Default: 1
    'SARIMA_MAX_P' : 2,             # maximum value of P (lag order - seasonal)                 | Default: 2
    'SARIMA_MAX_Q' : 2,             # maximum value of Q (order of moving average - seasonal)   | Default: 2
    'SARIMA_TEST' : 'adf',          # statistical test to use for determining stationarity
}
# Merge ensemble parameters (both models)
COMPARISON_PARAMS_DICT['ENSEMBLE'] = COMPARISON_PARAMS_DICT[SETTINGS['ENSEMBLE_MODEL']] + COMPARISON_PARAMS_DICT['LSTM']
SETTINGS['CELL_NAME'] = str(CELL).rsplit(sep='.', maxsplit=1)[-1].rsplit(sep="'", maxsplit=1)[0]

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------
def shuffle_in_unison(arrays):
    """
    Shuffle all arrays using the same random seed.

    Params:
        arrays (list): A list of arrays to shuffle.

    Returns:
        None
    """
    # shuffle all arrays using the same random seed
    state = np.random.get_state()
    for arr in range(len(arrays)):
        np.random.shuffle(arr)
        np.random.set_state(state)

def plot_candlestick_chart(data, window_size=1, title="Candlestick Chart", run_name="", test_name = ""):
    """
    Plot a candlestick chart for stock market financial data.

    Params:
        data (pd.DataFrame): The financial data with columns 'Open', 'High', 'Low', 'Close', and 'Date'.
        candlestick_window (int): The number of consecutive trading days to represent as a single candlestick.
        title (str): The title of the chart.

    Returns:
        None (displays the chart).
    """
    # Resample the data to create candlesticks representing 'candlestick_window' days.
    # Resample works by grouping the data by a certain time period and then aggregating the data
    # within each time period. In this case, we group by 'candlestick_window' days and aggregate
    # using the first, max, min, and last prices for each group.
    plot_data = data.resample(f'{window_size}D').agg({'Open': 'first',
                                                      'High': 'max',
                                                      'Low': 'min',
                                                      'Close': 'last'})

    # Create a custom style for the candlestick chart.
    custom_style = mpf.make_mpf_style(base_mpl_style="seaborn-darkgrid",
                                    gridcolor="white",
                                    facecolor="beige",
                                    edgecolor="black",)

    # Plot the candlestick chart.
    mpf.plot(plot_data, type='candle', style=custom_style,
             title=(title + f' (Window Size = {window_size})'),
             ylabel='Price', ylabel_lower='Date', show_nontrading=True,
             savefig = dict(fname=f'./logs/{test_name}/{run_name}/candlestick.jpg', dpi=300, bbox_inches='tight'))

def plot_boxplot_chart(data, window_size=2, title="Boxplot Chart", run_name="", test_name = "", label_skips=14):
    """
    Plot a boxplot chart for stock market financial data with a moving window.

    Params:
        data (pd.Series): The financial data (e.g., predicted prices) for a series of consecutive trading days.
        window_size (int): The size of the moving window for creating boxplots.
        title (str): The title of the chart.
        label_skips (int): The number of labels to skip between each displayed label.

    Returns:
        None (displays the chart).
    """
    print(data)
    # Calculate the number of boxplots to create.
    num_boxplots = len(data) - window_size + 1
    print(len([f'Day {i+1}-{i+window_size}' for i in range(num_boxplots)]))

    # Create a list to store data for each boxplot.
    plot_data = [data.values[i:i+window_size] for i in range(num_boxplots)]

    # Plot the boxplot chart.
    plt.figure(figsize=(14, 6))
    # Setting up the boxplot chart, including the labels for each boxplot.
    plt.title(title + f' (Window Size = {window_size})')
    plt.xlabel('Days')
    plt.ylabel('Prices')
    plt.xticks(rotation=65)
    plt.grid(True) # Adding a grid to the chart for better readability
    plt.boxplot(plot_data,
                labels=[f'Day {i+1}-{i+window_size}' for i in range(num_boxplots)],
                autorange=True,
                meanline=True,
                showmeans=True,)

    # Normally the x-axis labels are too close together to read with this many boxplots,
    # so here we can just disable every nth label to mitigate the overlap.
    for nth, lbl in enumerate(plt.gca().xaxis.get_ticklabels()):
        if nth % label_skips != 0:
            lbl.set_visible(False)
    plt.legend()
    plt.savefig(f'./logs/{test_name}/{run_name}/boxplot.jpg', dpi=300, bbox_inches='tight')
    if DISPLAY_PLOTS:
        plt.show()

def plot_ensemble(data, real_prices, predictions, run_name="", price_value="Close", test_name = ""):
    """
    Plot the test predictions including ensemble predictions.

    Params:
        data (pd.DataFrame): The financial data with columns 'Open', 'High', 'Low', 'Close', and 'Date'.
        real_prices (np.array): The actual closing prices for the test period.
        predictions (list): A list of tuples containing the predicted prices and the name of the model used to predict them.
    
    Returns:
        None (displays the chart).
    """
    # Align start and end dates of all predictions
    start_date = max(prediction[0].index[0] for prediction in predictions)
    end_date = min(prediction[0].index[-1] for prediction in predictions)
    # Remove any predictions on dates outside of this range
    predictions = [(prediction[0].loc[start_date:end_date], prediction[1]) for prediction in predictions]

    colours = ['blue', 'purple', 'green', 'orange', 'black']
    plot_error_percent = []
    plot_label = []
    for ii, prediction in enumerate(predictions):
        #prediction[0].fillna(inplace=True, method='bfill')
        plot_error_percent.append(prediction[0]['price_error_percent']*100)
        plot_label.append(prediction[1])
    plt.figure(figsize=(10, 12))
    plt.title(f'{SETTINGS["FORECAST_METHOD"]} | Test Prediction | Price Error (%)')
    plt.ylabel('Price Error (%)')
    plt.grid(True)
    plt.boxplot(plot_error_percent, labels=plot_label, autorange=True, meanline=True, showmeans=True)
    plt.legend()
    plt.savefig(f'./logs/{test_name}/{run_name}/price_error_percent.jpg', dpi=300, bbox_inches='tight')
    if DISPLAY_PLOTS:
        plt.show()

    # Plot the test prediction errors
    plt.figure(figsize=(12, 6))
    plt.title(f'{SETTINGS["FORECAST_METHOD"]} | Test Prediction | Price Error (USD)')
    plt.xlabel('Date')
    plt.ylabel('Price Error (USD)')
    plt.grid(True)
    data = data.sort_index()
    for ii, prediction in enumerate(predictions):
        plt.plot(prediction[0].index, prediction[0]['price_error'], label=prediction[1], color=colours[ii % len(colours)])
    plt.legend()
    plt.savefig(f'./logs/{test_name}/{run_name}/price_error.jpg', dpi=300, bbox_inches='tight')
    if DISPLAY_PLOTS:
        plt.show()

    # Plot the test predictions
    plt.figure(figsize=(12, 6))
    plt.title(f'{SETTINGS["FORECAST_METHOD"]} | Test Prediction | {price_value} (USD)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    #data = data.sort_index()
    for ii, prediction in enumerate(predictions):
        plt.plot(prediction[0].index, prediction[0][price_value], label=prediction[1], color=colours[ii % len(colours)])
    plt.plot(data.index, real_prices, label='Real Price', color='red')
    plt.legend()
    plt.savefig(f'./logs/{test_name}/{run_name}/test_prediction.jpg', dpi=300, bbox_inches='tight')
    if DISPLAY_PLOTS:
        plt.show()


def compare_runs(t_name, prm, plot_title, filename="loss_LSTM", pcol="loss", display_plots=True, label_skips=14, yaxis="Loss"):
    print(f"-- Comparing {t_name} Run | {plot_title} --")
    # Compare the loss history of different runs with a subplot for each run
    runs = pd.read_csv(f'./logs/{t_name}/training_runs.csv', index_col=0)

    plt.figure(figsize=(12, 6))
    fig = plt.get_current_fig_manager()
    fig.set_window_title(f'{t_name} - {plot_title}')
    style_cur = col_cur = 0
    style_list = ['-', '--', '-.', ':']
    col_list = ['r', 'g', 'b', 'k']

    # Really convoluted way of getting the list of parameters that we want to compare
    dyn_params = prm
    identical = runs[dyn_params].nunique() == 1
    diff = identical[identical == False].index.tolist()
    identical = identical[identical == True].index.tolist()
    
    for i, row in runs.iterrows():
        loss_history_path = f'./logs/{t_name}/{row["run_name"]}/{filename}.csv'
        if not os.path.exists(loss_history_path):
            print(f"Skipping {row['run_name']} as it does not have a loss history file")
            continue
        loss_history_df = pd.read_csv(loss_history_path, index_col=0)
        # Can ignore the first epoch since it's usually a lot higher than the rest
        loss_history_df = loss_history_df.iloc[1:]

        # change color for each run, only change style when color has been used
        run_differences = row[diff].to_dict()

        plt.plot(loss_history_df[pcol], label=f'{row["run_name"]} | {run_differences}', linestyle=style_list[style_cur%4], color=col_list[col_cur%4])
        col_cur += 1
        if col_cur%4 == 0:
            style_cur += 1

    label_list = runs[identical].to_dict(orient="records")[0]

    # Initialize an empty string to store the result
    label = ""
    # Format label string
    for i, (key, value) in enumerate(label_list.items()):
        label += f"{key}: {value}"
        # Add a newline character every 6 pairs (except for last)
        if (i + 1) % 6 == 0 and (i + 1) != len(label_list):
            label += '\n'
        else: label += ' | '

    for nth, lbl in enumerate(plt.gca().xaxis.get_ticklabels()):
        if nth % label_skips != 0:
            lbl.set_visible(False)
            plt.gca().xaxis.get_major_ticks()[nth].set_visible(False)
        else:
            lbl.set_visible(True)
            plt.gca().xaxis.get_major_ticks()[nth].set_visible(True)

    plt.ylabel(yaxis)
    plt.grid(True)
    plt.title(f"{plot_title}")
    plt.legend()
    plt.savefig(f"./logs/{t_name}/{filename}_{pcol}__comparison.jpg", dpi=300, bbox_inches="tight")
    if display_plots:
        plt.show()

def log_console_file(run_name, message, test_name=TEST_NAME):
    """
    Print a message to the console and save it to a log file.

    Params:
        run_name (str): The name of the run to save the message to.
        message (str): The message to print and save.
        test_name (str): The name of the test to save the message to.

    Returns:
        None
    """
    # Print the message to the console.
    print(message)

    # Check if the log file exists.
    if not os.path.exists(f'./logs/{test_name}/{run_name}/log.txt'):
        # Create the log file.
        os.makedirs(f'./logs/{test_name}/{run_name}', exist_ok=True)
        with open(f'./logs/{test_name}/{run_name}/log.txt', 'w', encoding='utf-8') as log_file:
            log_file.write(f'Log file for {test_name}/{run_name} created.\n---\n')

    # Save the message to a log file.
    with open(f'./logs/{test_name}/{run_name}/log.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(str(message) + '\n')

#------------------------------------------------------------------------------
# Load Data
#------------------------------------------------------------------------------
def load_data_and_process(company, data_source, predict_days,
                          train_start_date, train_end_date,
                          feature_columns, scale=True,
                          split_by_date=True, test_size=0.2,
                          shuffle=True, prediction_stepsize=1,
                          load_local=True, save_local=True,
                          local_path='./', resample=False,
                          resample_freq='D', pad_missing=True,
                          price_value="Close"):
    """
    Loads the data from the given company and processes it for training the model.

    Params:
        company (str/pd.DataFrame): the company you want to load, e.g. TSLA, AAPL, etc.
        start_date (str: 'YYYY-MM-DD'): start date of the training period
        end_date (str: 'YYYY-MM-DD'): end date of the training period
        predict_days (int): number of days to look back to base the prediction
        scale (bool): whether to scale the data
        split_by_date (bool): whether to split the data by date
        test_size (float): ratio for test data to training data
        shuffle (bool): whether to shuffle the data
        feature_columns (list): the list of features to use to feed into the model
        prediction_stepsize (int): the number of days to predict ahead
        load_local (bool): whether to load the data from local storage
        save_local (bool): whether to save the data to local storage
        local_path (str: './filepath_name'): the path to the local storage
        resample (bool): whether to include a copy of the data resampled to a lower frequency (resample_freq)
        resample_freq (str: 'D', 'W', 'M', etc.): the frequency to resample the data
        pad_missing (bool): whether to pad the missing dates with the last available value
        price_value (str: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'): the column to use as the target value

    Returns:
        result: a dict containing the following:
            df (pd.DataFrame): the original dataframe
            resampled_df (pd.DataFrame): the resampled dataframe
            column_scaler (dict): the MinMaxScaler instances for each feature
            last_sequence (np.array): the last sequence of the training data
            x_train (np.array): the training data
            y_train (np.array): the training labels
            x_test (np.array): the testing data
            y_test (np.array): the testing labels
            test_df (pd.DataFrame): the testing dataframe
            TRAIN_START (str): start date of the training data
            TRAIN_END (str): end date of the training data
            TEST_START (str): start date of the testing data
            TEST_END (str): end date of the testing data
    """
    # -------- Load the data --------
    # Check if data has been prepared before.
    if load_local and os.path.exists(os.path.join(local_path, company + '.csv')):
        # Load the saved data as a dataframe from local storage
        data = pd.read_csv(os.path.join(local_path, company + '.csv'), parse_dates=True, index_col=0)
    else:
        # Download the data as a dataframe
        if isinstance(company, str):
            if data_source == "yahoo":
                data = yf.download(company, start=train_start_date, end=train_end_date, progress=False)
            else:
                raise ValueError("'data_source' str value can only be 'yahoo' for now")

            # Ensure that the data is a dataframe
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Downloaded data type is not a valid dataframe")

        # Use the data as a dataframe
        elif isinstance(company, pd.DataFrame):
            data = company
        else:
            raise TypeError("'company' variable must be either: str or `pd.DataFrame`")

        # -------- Save the data ----

        # Save the data into a directory
        if save_local:
            # Create the directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            # Save the data to local storage as a dataframe
            data.to_csv(os.path.join(local_path, company + '.csv'))

    result = {} # Storing result in a dict for convenience

    # We then pad the missing dates with the last available value, since we can't drop the missing dates if
    # we want to preserve the periodicity and chronological order of the data.
    if pad_missing:
        data = data.ffill()

    # Since our data is not perfectly periodic (with some missing dates), we need to either
    # resample the data to a lower frequency or drop the missing dates, however we can't drop the
    # missing dates since we need the data to be periodic and in chronological order for SARIMA and ARIMA.
    # Therefore, we resample the data to a lower frequency, and use the mean of the values for each week.
    if resample:
        resampled_data = data.resample(resample_freq).mean()
        result['resampled_df'] = resampled_data.copy() # include resampled dataframe copy too

    #data["date"] = data.index
    result['df'] = data.copy() # include original dataframe copy too

    # -------- Process the data --------

    # Make sure that all feature_columns exist in the dataframe
    for col in feature_columns:
        # Will throw an assert error if the column does not exist
        assert col in data.columns, f"'{col}' does not exist in the dataframe."

    # Add date as a column
    if "date" not in data.columns:
        # ensuring we have a stored DatetimeIndex for the dataframe
        data["date"] = data.index

    # Plot the close price of the stock, to ensure our data is correct
    # mpf.plot(data, type='candle', style='yahoo', volume=True, mav=(20), title=f'{COMPANY} Stock Price')

    if scale:
        # We create a new column for each feature, with the suffix "_unscaled"
        # to store the unscaled data.
        column_scaler = {}
        # Scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = MinMaxScaler()
            # Using np.expand_dims to reshape the data from (x,) to (x,1)
            # in order to fit MinMaxScaler, which requires a 2D array.
            # (much more elegant than .reshape(-1,1), more akin to pytorch unsqueeze())
            # https://note.nkmk.me/en/python-numpy-newaxis/

            # We then scale the data and store it in the original column.
            data[column] = scaler.fit_transform(np.expand_dims(data[column].values, axis=1))
            column_scaler[column] = scaler

        # Add the MinMaxScaler instances to the result returned, this will allow us to
        # convert back to actual prices later.
        result["column_scaler"] = column_scaler

    # Add the target column (label) by shifting by `prediction_stepsize`
    # Target column is the future Close price shifted by `prediction_stepsize` days
    data['future'] = data[price_value].shift(-prediction_stepsize)
    # Here we add a new column "future" to the DataFrame, shifted by prediction_stepsize
    # days into the future (aka the target 'future stock price' we want to predict)

    # last_sequence will be used for predicting future stock prices.
    last_sequence = np.array(data[feature_columns].tail(prediction_stepsize))
    # (tail(prediction_stepsize) returns the last prediction_stepsize rows of the dataframe)
    # last_sequence is assigned the last prediction_stepsize rows of feature_columns.

    # last `prediction_stepsize` columns of feature_columns contains NaN in future column
    data.dropna(inplace=True)
    # Rows with NaN values are dropped from the dataframe using dropna()
    # (inplace=True means that the dataframe is modified in place, rather than a copy being returned)

    # Using a deque to store the sequences, which is a list-like container
    # The advantage here is that we can specify a maxlen, which will automatically
    # remove the oldest element when maxlen is reached (aka beyond the lookback period)
    sequence_data = []
    sequences = deque(maxlen=predict_days)

    # This loop iterates through the DataFrame rows, creating sequences
    # of length predict_days from the feature_columns data and appending
    # the corresponding future target value.
    for entry, target in zip(data[feature_columns + ["date"]].values, data['future'].values):
        # zip() returns an iterator of tuples, in this case we are zipping the
        # feature_columns and the 'future' (target) column for each row in the dataframe
        # (So each tuple is a a pair of (feature_columns, 'future'))
        sequences.append(entry)
        # We then append each tuple to the sequences deque
        # Once the deque reaches maxlen, we append the sequence to sequence_data
        # (removing the oldest element from the deque)
        # This should efficiently create a list of sequences of length predict_days
        if len(sequences) == predict_days:
            # Sequences and targets are stored in the sequence_data list.
            sequence_data.append([np.array(sequences), target])
            # DatetimeIndex is stored in the dates list, and used to select the
            # corresponding rows from the original DataFrame later (splitting test and training data)


    # Here we prepare the final sequence for predicting future stock prices,
    # by combining all remaining entries from the deque sequences and the last
    # prediction_stepsize rows of feature_columns data to create the last sequence.
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    # The last_sequence constructed now is a list of lists, with each inner list
    # containing the data for the feature columns.
    last_sequence = np.array(last_sequence).astype(np.float32)
    # We then convert last_sequence list of lists into a multidimensional NumPy array.
    # The .astype(np.float32) conversion is used to ensure that the array elements are
    # of data type float32, which is commonly used in machine learning models
    # (and done here for consistency).
    result['last_sequence'] = last_sequence # add to result

    # Construct the training data sequence and corresponding labels from sequence_data
    train_x, train_y = [], []
    for seq, target in sequence_data:
        train_x.append(seq)
        train_y.append(target)
    # Convert to numpy arrays
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # If split_by_date is True, the dataset is split into
    # training and testing sets based on dates.
    # The ratio of training to testing data is determined by test_size.
    if split_by_date:
        train_samples = int((1 - test_size) * len(train_x))
        result["x_train"] = train_x[:train_samples]
        result["y_train"] = train_y[:train_samples]
        result["x_test"]  = train_x[train_samples:]
        result["y_test"]  = train_y[train_samples:]

        result["x_train"] = train_x[:train_samples]
        result["y_train"] = train_y[:train_samples]
        result["x_test"]  = train_x[train_samples:]
        result["y_test"]  = train_y[train_samples:]
        # The training and testing datasets are stored in result["x_train"] and result["x_test"].
        if shuffle: # training and testing datasets are shuffled.
            # Using function shuffle_in_unison() borrowed from P1
            shuffle_in_unison((result["x_train"], result["y_train"]))
            shuffle_in_unison((result["x_test"], result["y_test"]))

        if resample:
            resampled_train_samples = int((1 - test_size) * len(result["resampled_df"]))
            result["resampled_test"] = result["resampled_df"].iloc[resampled_train_samples:]

    # result["x_train"] and result["x_test"] are three-dimensional arrays, where the second
    # dimension contains the sequences and the third dimension holds the features and the date.


    # get the list of test set dates
    dates = result["x_test"][:, -1, -1]

    # Since the third dimension of result["x_test"] holds the date information for each sequence,
    # we can extract the date of each test sequence with [:, -1, -1] and store it in an array 'dates'.

    
    result["test_df"] = result["df"].loc[dates]

    # We then use the dates array to select the corresponding rows from the original DataFrame
    # and store the resulting DataFrame in result["test_df"].
    # Note that the dates array contains duplicate dates, so the resulting DataFrame will also
    # contain duplicate rows. We can remove these duplicate rows later.

    # result["df"] holds the original DataFrame containing the entire dataset.
    # we can use the extracted dates array to index and select rows from the original
    # DataFrame, effectively creating a new DataFrame containing only the rows
    # corresponding to the testing data dates. (resulting DataFrame then stored in result["test_df"].)

    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    # Here we remove duplicated rows from the testing DataFrame while retaining the
    # first occurrence of each duplicated date.
    # The resulting DataFrame without duplicated dates is stored back in result["test_df"].

    # We'll include a start and end date for the training and testing sets in the result dict.
    result['TRAIN_START'] = result["x_train"][:, -1, -1][0]
    result['TRAIN_END'] = result["x_train"][:, -1, -1][-1]
    result['TEST_START'] = result["x_test"][:, -1, -1][0]
    result['TEST_END'] = result["x_test"][:, -1, -1][-1]

    # Lastly, we modify the training and testing data arrays stored in result to ensure
    # that they contain only the feature columns and not the date information.
    result["x_train"] = result["x_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["x_test"] = result["x_test"][:, :, :len(feature_columns)].astype(np.float32)
    # As with before, we can use slice notation with [:, :, :len(feature_columns)] to effectively
    # remove the date information from each array, leaving only the feature columns.

    return result # Return the result dict containing the processed data

loaded_data = load_data_and_process(
    company             = SETTINGS['COMPANY'],
    data_source         = SETTINGS['DATA_SOURCE'],
    predict_days        = SETTINGS['PREDICT_INCR_PAST'],
    train_start_date    = SETTINGS['TRAIN_START'],
    train_end_date      = SETTINGS['TRAIN_END'],
    feature_columns     = SETTINGS['FEATURE_COLUMNS'],
    scale               = SETTINGS['SCALE_FEATURES'],
    split_by_date       = SETTINGS['SPLIT_BY_DATE'],
    test_size           = SETTINGS['TEST_SIZE'],
    shuffle             = SETTINGS['SHUFFLE_DATASET'],
    prediction_stepsize = SETTINGS['PREDICT_INCR_AHEAD'],
    load_local          = SETTINGS['LOCAL_LOAD'],
    save_local          = SETTINGS['LOCAL_SAVE'],
    local_path          = SETTINGS['LOCAL_PATH'],
    resample            = SETTINGS['RESAMPLE_DATASET'],
    price_value         = SETTINGS['PRICE_VALUE'],
    resample_freq       = SETTINGS['RESAMPLE_FREQ'],
    pad_missing         = SETTINGS['PAD_MISSING_DAYS'])

SETTINGS['TRAIN_START'] = loaded_data['TRAIN_START']
SETTINGS['TRAIN_END'] = loaded_data['TRAIN_END']
SETTINGS['TEST_START'] = loaded_data['TEST_START']
SETTINGS['TEST_END'] = loaded_data['TEST_END']

#------------------------------------------------------------------------------
# Build the Model
#------------------------------------------------------------------------------

# Create the function to build the model (LSTM)
def create_model(n_feats, seq_len=60, units=50, cell=LSTM, n_lays=2, dropout=0.2,
                loss="mean_absolute_error", optim="adam"):
    """
    Creates the model using the given parameters.
    
    Params:
        n_feats (int): the number of features to use to feed into the model. | len(FEATURE_COLUMNS)
        seq_len (int): number of days to look back to base the prediction | Default: 60
        units (int): the number of units/nodes in each layer | Default: 50
        cell (LSTM/GRU): the type of cell to use | 'LSTM', 'GRU', 'LSTM', 'RNN', 'CNN' | Default: LSTM
        n_lays (int): the number of layers | Default: 2
        dropout (float): the dropout rate defines the fraction of deactivated neurons, eg; 0.2 == 20%.
        loss (str): the loss function | 'mean_absolute_error', 'huber_loss', 'cosine_similarity' | Default: "mean_absolute_error"
        optim (str): the optimizer function | 'adam', 'rmsprop', 'sgd', 'adadelta' | Default: "adam"

    Returns:
        model (tf.keras.src.engine.sequential.Sequential): the model
    """
    created_model = Sequential()# Basic neural network
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # for some useful examples
    for layer in range(n_lays):
        if layer == 0:
            # first layer
            created_model.add(cell(units, return_sequences=True, batch_input_shape=(None, seq_len, n_feats)))
            # This is our first hidden layer which also spcifies an input layer.
            # That's why we specify the input shape for this layer;
            # i.e. the format of each training example
            # The above would be equivalent to the following two lines of code:
            # model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
            # model.add(LSTM(units=50, return_sequences=True))
            # For advanced explanation of return_sequences:
            # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
            # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
            # As explained there, for a stacked LSTM, you must set return_sequences=True
            # when stacking LSTM layers so that the next LSTM layer has a
            # three-dimensional sequence input.

            # Finally, units specifies the number of nodes in this layer.
            # This is one of the parameters you want to play with to see what number
            # of units will give you better prediction quality (for your problem)
        elif layer == n_lays - 1:
            # last layer
            created_model.add(cell(units=units, return_sequences=False))
        else:
            # hidden layers
            created_model.add(cell(units=units, return_sequences=True))
            # More on Stacked LSTM:
            # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        # add dropout after each layer
        created_model.add(Dropout(dropout))
        # The Dropout layer randomly sets input units to 0 with a frequency of
        # rate (= 0.2 above) at each step during training time, which helps
        # prevent overfitting (one of the major problems of ML).
    created_model.add(Dense(units=1, activation="linear"))
    # Prediction of the next closing value of the stock price

    # We compile the model by specify the parameters for the model
    # See lecture Week 6 (COS30018)
    created_model.compile(loss=loss,
                          metrics=[loss],
                          optimizer=optim)
    # The loss function is a measure of how good a prediction model does in terms
    # of being able to predict the expected outcome.
    return created_model

# Creating our own callback to be able to save and store the loss history
class LossCallback(tf.keras.callbacks.Callback):
    """
    A callback class that records the loss history of a Keras model during training.

    Attributes:
        loss_history (dict): A dictionary that stores the loss history of the model.
    """

    def __init__(self):
        super().__init__()
        self.loss_history = {'loss': {}}

    def on_epoch_end(self, epoch, logs=None):
        """
        A function that is called at the end of each epoch during training.

        Params:
            epoch (int): The current epoch number.
            logs (dict): A dictionary that contains the training and validation metrics for the current epoch.
        """
        # We append the loss metric to the history dict
        self.loss_history["loss"][epoch] = logs["loss"]

# ---------------------------------------------------------------------
# Training LSTM model
# ---------------------------------------------------------------------
def run_model(settings):
    """
    Runs the model using the given parameters.

    Params:
        settings (dict): the dict containing the settings for the model
    
    Returns:
        time_ended (str): the time the model finished running
    """
    # Make directory if it doesn't exist
    if not os.path.exists(os.path.join(f'./logs/{settings["TEST_NAME"]}/')):
        os.makedirs(os.path.join(f'./logs/{settings["TEST_NAME"]}/'))
    if not os.path.exists(os.path.join(f'./logs/{settings["TEST_NAME"]}/')):
        os.makedirs(os.path.join(f'./logs/{settings["TEST_NAME"]}/'))

    history_callback = LossCallback()

    # Call the function to create our model - Model1 is our LSTM model, Model2 is ARIMA (our ensemble model)
    model1 = create_model(
        n_feats = len(settings['FEATURE_COLUMNS']), # number of features to use to feed into the model
        seq_len = settings['PREDICT_INCR_PAST'], # number of days to look back to base the prediction
        units   = settings['UNITS'], # number of units in each layer
        cell    = CELL, # the type of cell to use
        n_lays  = settings['NUM_LAYERS'], # the number of layers
        # Dropout is a regularization technique to help prevent overfitting by randomly deactivating
        # neurons during training to encourage the network to learn more robust and generalizable
        # features, since the network is penalized for over-relying on any particular neuron.
        # Dropout variable is the fraction of the neurons that are deactivated, so 0.2 == 20%.
        dropout = settings['DROPOUT'], # the dropout rate
        loss    = settings['LOSS'], # the loss function
        optim   = settings['OPTIMIZER']) # the optimizer function

    # Get run start time:
    time_started = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Fit the model
    model1.fit(loaded_data['x_train'], loaded_data['y_train'], epochs=settings['EPOCHS'], batch_size=settings['BATCH_SIZE'], callbacks=[history_callback])

    # Get run end time:
    time_ended = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{settings['COMPANY']}_-_{time_ended}"
    run_filepath = f'./logs/{settings["TEST_NAME"]}/{run_name}'
    log_console_file(run_name, f"Model1 fitting finished at: {time_ended}, setting run_name to: {run_name}")
    log_console_file(run_name, model1.summary()) # Print model summary

    # Create a directory for the run
    if not os.path.exists(os.path.join(f"./logs/{settings['TEST_NAME']}/", run_name)):
        os.makedirs(os.path.join(f"./logs/{settings['TEST_NAME']}/", run_name))

    # Access loss_history dict after training
    loss_history = history_callback.loss_history

    # Plot the loss history
    if PLOT_LOSS:
        # Extract the loss values and epoch numbers (except first epoch)
        epochs = list(loss_history["loss"].keys())[1:]
        loss_values = list(loss_history["loss"].values())[1:]

        if SAVE_LOSS_PLOT or DISPLAY_PLOTS:
            # Create a plot
            plt.figure(figsize=(12, 6))
            fig = plt.get_current_fig_manager()
            fig.set_window_title(f'Model: {settings["COMPANY"]} Stock Price Prediction | ./{run_name}/loss_plot.jpg')
            plt.plot(epochs, loss_values, marker='o', linestyle='-')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.title(f'Model: {settings["COMPANY"]} Stock Price Prediction | \
            Total Epochs : {str(settings["EPOCHS"])}\n\
            Batch Size: {str(settings["BATCH_SIZE"])} | \
            Layer Count: {str(settings["NUM_LAYERS"])} |\
            Units (per Layer): {str(settings["UNITS"])} |\
            Dropout: {str(settings["DROPOUT"])}\n\
            Cell: {settings["CELL_NAME"]} |\
            Optimizer: {str(settings["OPTIMIZER"])} |\
            Loss Function: {str(settings["LOSS"])}')
            plt.ylabel('Training Loss')
            plt.xlabel('Epoch')
            plt.legend(['Loss'])

            # Save before display since display will clear the plot
            if SAVE_LOSS_PLOT:
                plt.savefig(f'{run_filepath}/loss_plot.jpg', dpi=300, bbox_inches='tight')
            # Display the plot
            if DISPLAY_PLOTS:
                plt.show()

        ## Save loss history
        loss_df = pd.DataFrame(loss_history)
        # Save the dataframe to CSV
        loss_df.to_csv(f'{run_filepath}/loss_LSTM.csv')

    # Save model
    model1.save(f'{run_filepath}/lstm.keras')

    settings['run_name'] = run_name
    settings['time_started'] = time_started
    settings['time_ended'] = time_ended


    # ------------------------ ARIMA AND SARIMA ------------------------

    # ARIMA AND SARIMA
    # ARIMA and SARIMA are statistical models that use time series data to predict future values.
    # ARIMA: AutoRegressive Integrated Moving Average | SARIMA: Seasonal AutoRegressive Integrated Moving Average
    # The major difference between ARIMA and SARIMA is that SARIMA takes into account the seasonality
    # of the data, which is useful for our stock price prediction problem since stock prices are highly seasonal

    # ARIMA - For non-seasonal data
    # p, d, and q are the hyperparameters of the ARIMA model.
    # p is the number of lag observations included in the model, also called the lag order.
    # d is the number of times that the raw observations are differenced, also called the degree of differencing.
    # q is the size of the moving average window, also called the order of moving average.

    # We could use a resampled version of the data for ARIMA and SARIMA, since our data is not perfectly periodic,
    # and ARIMA and SARIMA require the data to be periodic and in chronological order.
    sarima_train_con = loaded_data['resampled_df'][settings['PRICE_VALUE']]

    # Plot the decomposed time series to see the trend, seasonality, and residuals
    plt.rc('figure',figsize=(12,8))
    plt.rc('font',size=15)
    decomp_result = seasonal_decompose(sarima_train_con, model = 'additive', period = settings['LAGS_VAL'])
    fig = decomp_result.plot()
    plt.legend()
    plt.savefig(f'{run_filepath}/SARIMA_decomp_plot.jpg', dpi = 300, bbox_inches='tight')

    # Plot the ACF and PACF plots. These can be used to determine the hyperparameters p, d, and q.
    # The ACF plot (Autocorrelation Function) shows the correlation of the series with itself,
    # lagged by x time units. The PACF plot (Partial Autocorrelation Function) shows the correlation
    # between the series and its lag, after removing the effect of previous lags.
    fig1 = plot_acf(sarima_train_con, lags = settings["LAGS_VAL"])
    plt.title(f'ACF Plot for {settings["COMPANY"]} Stock Price')
    fig1.savefig(f'{run_filepath}/SARIMA_acf_plot.jpg', dpi = 300, bbox_inches = 'tight')
    fig2 = plot_pacf(sarima_train_con, lags = settings["LAGS_VAL"])
    plt.title(f'PACF Plot for {settings["COMPANY"]} Stock Price')
    fig2.savefig(f'{run_filepath}/SARIMA_pacf_plot.jpg', dpi = 300, bbox_inches = 'tight')

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------
    # Load the test data
    test_data = loaded_data['test_df']

    actual_close_prices = loaded_data['test_df'][settings['PRICE_VALUE']]

    # We need to do the following because to predict the closing price of the first PREDICTION_DAYS of the
    # test period [loaded_data['TEST_START'], loaded_data['TEST_END']], we'll need the data from the training period
    total_dataset = pd.concat((loaded_data['df'][settings['FEATURE_COLUMNS']], test_data[settings['FEATURE_COLUMNS']]), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - settings["PREDICT_INCR_PAST"]:].values

    # Normalize the data within each column using the column scaler value
    for t, feat in enumerate(settings['FEATURE_COLUMNS']):
        model_inputs[:, t] = loaded_data["column_scaler"][feat].transform(model_inputs[:, t].reshape(-1, 1)).reshape(-1)

    #------------------------------------------------------------------------------
    # Predict on Test Data
    #------------------------------------------------------------------------------
    # test_df:      (240, 6)        | 2020-01-01 to 2020-12-31 | Real Values       | Open, High, Low, Close, Adj Close, Volume
    # df:           (1258, 6)       | 2016-01-01 to 2020-12-31 | Real Values       | Open, High, Low, Close, Adj Close, Volume
    # model_inputs: (300, 6)        | 2020-01-01 to 2020-12-31 | Normalized Values |
    # x_test:       (240, 60, 5)    | 2020-01-01 to 2020-12-31 | Normalized Values |
    # y_test:       (240,)          | 2020-01-01 to 2020-12-31 | Normalized Values |
    # y_train:      (998,)          | 2016-01-01 to 2019-12-31 | Normalized Values |
    # x_train:      (998, 60, 5)    | 2016-01-01 to 2019-12-31 | Normalized Values |

    # -- ARIMA
    model2 = sm.tsa.ARIMA(
        sarima_train_con, # Create ARIMA model with Close price as target
        order=(settings["ARIMA_P"], settings["ARIMA_D"], settings["ARIMA_Q"]), # p, d, and q hyperparameters
        freq= settings["RESAMPLE_FREQ"]) # Resample frequency

    # Fit the model
    arima_results = model2.fit()
    log_console_file(run_name, arima_results.summary())

    arima_results.plot_diagnostics(figsize = (16, 8))
    plt.legend()
    plt.savefig(f'{run_filepath}/ARIMA_diagnostics.jpg', dpi = 300, bbox_inches = 'tight')
    if DISPLAY_PLOTS:
        plt.show()

    # -- SARIMA
    # Perform grid search to find the best hyperparameters for the SARIMA model
    model3 = pm.auto_arima(
        y = sarima_train_con, test = settings['SARIMA_TEST'],
        start_p = settings['SARIMA_START_p'], start_q = settings['SARIMA_START_q'],
        start_P = settings['SARIMA_START_P'], start_Q = settings['SARIMA_START_Q'],
        max_p   = settings['SARIMA_MAX_p'], max_q = settings['SARIMA_MAX_q'],
        max_P   = settings['SARIMA_MAX_P'], max_Q = settings['SARIMA_MAX_Q'],
        max_d   = settings['SARIMA_MAX_d'], max_D = settings['SARIMA_MAX_D'],
        seasonal = settings['SARIMA_SEASONAL'], m = settings['SEASONAL_PERIOD'],
        d = settings['SARIMA_d'], D = settings['SARIMA_D'], stepwise=True,
        trace=False, error_action='ignore', suppress_warnings=True)

    # Print the best model's hyperparameters
    log_console_file(run_name, f"Best SARIMA Model: ({model3.order}), ({model3.seasonal_order})")

    # Save the best SARIMA settings to the settings dictionary
    settings['sarima_order'] = str(model3.order)
    settings['sarima_seasonal_order'] = str(model3.seasonal_order)

    settings['arima_order'] = str(model2.order)

    # Train the SARIMA model
    sarima_results = model3.fit(sarima_train_con)

    # Print the SARIMA model summary
    log_console_file(run_name, sarima_results.summary())

    # Plot the SARIMA model diagnostics and save the figure
    sarima_results.plot_diagnostics(figsize=(16, 8))
    plt.legend()
    plt.savefig(f'{run_filepath}/SARIMA_diagnostics.jpg', dpi=300, bbox_inches='tight')

    # Define the forecast index as a range of dates from the start to the end of the test set
    forecast_index = pd.date_range(loaded_data['TEST_START'], loaded_data['TEST_END'], freq = settings['RESAMPLE_FREQ'])

    # If we remove the testing data from the training data, we can test the ARIMA model out-of-sample capabilities
    if settings['ARIMA_OOS_FORECASTS'] > 0:
        sarima_train_con = sarima_train_con[:-settings['ARIMA_OOS_FORECASTS']]

    elif settings['ARIMA_OOS_FORECASTS'] == -1:
        if settings['FORECAST_METHOD'] == 'speculative':
            sarima_train_con = sarima_train_con.loc[:loaded_data['TEST_START']]

    # Copy the SARIMA training data to a new variable for dynamic forecasting
    sarima_train_dyn = sarima_train_con.copy()

    log_console_file(run_name, f'Refitting ARIMA model to {sarima_train_dyn.index[0]} to {sarima_train_dyn.index[-1]}')

    # Define the columns for the forecasts dataframe
    cols = ['ARIMA', 'SARIMA']

    # Create a new dataframe to store the forecasts
    forecasts = pd.DataFrame(index=forecast_index, columns=cols)
    if settings['USE_SARIMA'] or settings['USE_ARIMA']:
        log_console_file(run_name, '------------------\nBEGINNING ARIMA/SARIMA FORECASTING\n------------------')
                
        # Loop through the date range
        for date_i in range(0, len(forecast_index)):
            # Remove dates yet to be forecasted from the SARIMA training data
            if settings['ARIMA_OOS_FORECASTS'] != 0:
                if settings['ARIMA_OOS_FORECASTS'] > 0:
                    sarima_train_dyn = sarima_train_dyn.iloc[:-settings['ARIMA_OOS_FORECASTS']]

                elif settings['ARIMA_OOS_FORECASTS'] == -1:
                    if settings['FORECAST_METHOD'] == 'speculative':
                        sarima_train_dyn = sarima_train_dyn.loc[:forecasts.index[date_i]]
                        print(forecasts.index[date_i])
                        
                    else:
                        sarima_train_dyn = sarima_train_con.loc[:forecasts.index[date_i]]

                if settings['USE_ARIMA']:
                    model2 = sm.tsa.ARIMA(
                        sarima_train_dyn, # Create ARIMA model with Close price as target
                        order = (settings["ARIMA_P"], settings["ARIMA_D"], settings["ARIMA_Q"]), # p, d, and q hyperparameters
                        freq = settings["RESAMPLE_FREQ"]) # Resample frequency
                    arima_results = model2.fit()

            
            # Get the start and end dates for the forecast period
            start_fc = forecasts.index[date_i]
            log_console_file(run_name, f' --- Dataset Period: {sarima_train_dyn.index[0]} to {sarima_train_dyn.index[-1]} ---')
            log_console_file(run_name, f'      - Forecasting: {start_fc}')
            if settings['USE_ARIMA']:
                # Forecast the next PREDICT_INCR_AHEAD days using ARIMA and SARIMA models
                # ARIMA fit and predict
                fcast_arima = arima_results.get_prediction(start=start_fc, end=start_fc)

            # SARIMA fit and predict
            if settings['USE_SARIMA']: 
                if settings['FORECAST_METHOD'] == 'speculative':
                    fcast_sarima = sarima_results.fit_predict(y=sarima_train_dyn, n_periods=1)
                else:
                    fcast_sarima = sarima_results.fit_predict(y=sarima_train_dyn.iloc[:1], n_periods=1)

                # Convert the SARIMA forecast to a pandas series with the forecast index
                sarima_fcast_series = pd.Series(fcast_sarima, index=fcast_sarima.index)

                # Print the SARIMA training data before and after concatenation (for debugging)
                if settings['FORECAST_METHOD'] == 'speculative':
                    sarima_train_dyn = sarima_fcast_series.combine_first(sarima_train_dyn)

            # Add the ARIMA and SARIMA forecasts to the forecasts dataframe
            if settings['USE_ARIMA']:
                forecasts.loc[start_fc:start_fc, 'ARIMA'] = fcast_arima.predicted_mean
                log_console_file(run_name, f'         - ARIMA  - ({fcast_arima.predicted_mean.index[0]}): {fcast_arima.predicted_mean.values}')
            if settings['USE_SARIMA']:
                forecasts.loc[start_fc:start_fc, 'SARIMA'] = sarima_fcast_series
                log_console_file(run_name, f'         - SARIMA - ({sarima_fcast_series.index[0]}): {sarima_fcast_series.values}')

        log_console_file(run_name, '------------------\nCOMPLETED ARIMA/SARIMA FORECASTING\n------------------')
        forecasts.fillna(method='ffill', inplace=True)
        # Print the forecasts dataframe (for debugging)
        #log_console_file(run_name, f'------\nForecasts: {forecasts}\n------\n')

    # Number of predictions to make
    x_test = []
    
    # Loop through the past PREDICT_INCR_PAST days
    for x in range(settings["PREDICT_INCR_PAST"], len(model_inputs)):
        # Append the past PREDICT_INCR_PAST days to x_test, each containing corresponding FEATURE_COLUMNS
        x_test.append(model_inputs[x - settings["PREDICT_INCR_PAST"]:x, 0:len(settings['FEATURE_COLUMNS'])])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Predict prices using LSTM model
    real_predicted_prices_lstm = model1.predict(x_test)

    # Create a dataframe to store the predicted prices
    predicted_prices_lstm = pd.DataFrame(index=test_data.index, columns=[settings['PRICE_VALUE']])

    # Inverse transform the predicted prices and add them to the dataframe
    for feat in settings['FEATURE_COLUMNS']:
        predicted_prices_lstm[feat] = loaded_data["column_scaler"][feat].inverse_transform(real_predicted_prices_lstm).reshape(-1)

    if settings['USE_SARIMA'] and settings['USE_ARIMA']:
        # Plot the forecasts
        _, axis = plt.subplots(figsize=(16, 10))
        axis.plot(loaded_data['df'][settings['PRICE_VALUE']], label='Training data')
        axis.plot(test_data[settings['PRICE_VALUE']], label='Testing data')
        axis.plot(forecasts['ARIMA'], label='ARIMA forecast')
        axis.plot(forecasts['SARIMA'], label='SARIMA forecast')
        axis.plot(predicted_prices_lstm[settings['PRICE_VALUE']].loc[loaded_data['TEST_START'] + dt.timedelta(days=1):], label='LSTM forecast')
        axis.set_xlabel("Date")
        axis.set_ylabel("Price")
        # Rotate the x-axis labels by 45 degrees for readability
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
        plt.title('ARIMA & SARIMA Price Prediction')
        plt.legend()
        plt.savefig(f'{run_filepath}/arima_sarima_comparison.jpg', dpi=300, bbox_inches='tight')


    # Create a date range for the forecast period
    forecast_index = pd.date_range(loaded_data['TEST_START'], loaded_data['TEST_END'], freq=settings['RESAMPLE_FREQ'])

    # Create dataframes for the predicted prices from ARIMA and SARIMA models
    predicted_prices_arima = pd.DataFrame(index=forecasts.index)
    predicted_prices_arima[settings['PRICE_VALUE']] = forecasts['ARIMA'].values

    predicted_prices_sarima = pd.DataFrame(index=forecasts.index)
    predicted_prices_sarima[settings['PRICE_VALUE']] = forecasts['SARIMA'].values


    # Calculate the error and error percentage for the predicted prices from the LSTM model
    predicted_prices_lstm['price_error'] = predicted_prices_lstm[settings['PRICE_VALUE']] - actual_close_prices.values
    predicted_prices_lstm['price_error_percent'] = predicted_prices_lstm['price_error'] / actual_close_prices.values

    actual_resampled_close_prices = loaded_data['resampled_test'][settings['PRICE_VALUE']].loc[loaded_data['TEST_START']:loaded_data['TEST_END']]

    # Calculate the error and error percentage for the predicted prices from the ARIMA model
    predicted_prices_arima['price_error'] = forecasts['ARIMA'].values - actual_resampled_close_prices
    predicted_prices_arima['price_error_percent'] = predicted_prices_arima['price_error'] / actual_resampled_close_prices

    # Calculate the error and error percentage for the predicted prices from the SARIMA model
    predicted_prices_sarima['price_error'] = forecasts['SARIMA'].values - actual_resampled_close_prices
    predicted_prices_sarima['price_error_percent'] = predicted_prices_sarima['price_error'] / actual_resampled_close_prices

    # Create an empty dataframe for the ensemble model predictions
    predicted_prices_ensemble = pd.DataFrame(index=test_data.index)#forecasts.index)

    # Ensemble | Average prediction between SARIMA/ARIMA and LSTM model
    # ARIMA and SARIMA outputs are in a different frequency than the LSTM model,
    # so we need to resample to daily frequency before averaging them with the LSTM model
    predicted_prices_ensemble[settings['PRICE_VALUE']] = (predicted_prices_lstm[settings['PRICE_VALUE']] + predicted_prices_sarima[settings['PRICE_VALUE']].resample('D').interpolate(method='pad')) / 2#interpolate(method='linear').ffill()) / 2
    
    # Fill both ends for any NaN values (likely a better way to do this)
    predicted_prices_ensemble.fillna(method='bfill', inplace=True)
    predicted_prices_ensemble.fillna(method='ffill', inplace=True)#.dropna(inplace=True)
    
    # Calculate the error and error percentage for the predicted prices from the ensemble model
    predicted_prices_ensemble['price_error'] = predicted_prices_ensemble[settings['PRICE_VALUE']] - actual_close_prices.loc[predicted_prices_ensemble.index[0]:predicted_prices_ensemble.index[-1]].values
    predicted_prices_ensemble['price_error_percent'] = predicted_prices_ensemble['price_error'] / actual_close_prices.loc[predicted_prices_ensemble.index[0]:predicted_prices_ensemble.index[-1]].values

    # Log the predicted prices and errors for each model to the console
    log_console_file(run_name, f'------------------\n - LSTM Prediction Sample:\n{predicted_prices_lstm}\n------------------\n')
    log_console_file(run_name, f'------------------\n - ARIMA Prediction Sample:\n{predicted_prices_arima}\n------------------\n')
    log_console_file(run_name, f'------------------\n - SARIMA Prediction Sample:\n{predicted_prices_sarima}\n------------------\n')
    log_console_file(run_name, f'------------------\n - ENSEMBLE Prediction Sample:\n{predicted_prices_ensemble}\n------------------\n')

    #------------------------------------------------------------------------------
    # Save predictions
    #------------------------------------------------------------------------------

    # Save settings to a JSON file in the run directory
    save_settings(settings, f'{run_filepath}/settings.json')

    # Convert settings dictionary to a DataFrame (via str first to avoid errors)
    settings_df = {key: str(value) for key, value in settings.items()}
    settings_df = pd.DataFrame.from_dict(settings_df, orient='index')
    settings_df = settings_df.transpose()

    # Check if the training_runs.csv file exists
    if not os.path.exists(f'./logs/{settings["TEST_NAME"]}/training_runs.csv'):
        # If it doesn't exist, create the file and write the settings DataFrame to it
        settings_df.to_csv(f'./logs/{settings["TEST_NAME"]}/training_runs.csv', index_label='Run #')
    else:
        # If it does exist, append the settings DataFrame to the file
        settings_df.to_csv(f'./logs/{settings["TEST_NAME"]}/training_runs.csv', index_label='Run #', mode='a', header=False)

    ## Save predictions to CSV
    # Convert predicted prices to DataFrames and save them to CSV files
    pd.DataFrame(predicted_prices_lstm).to_csv(f'{run_filepath}/LSTM.csv')
    pd.DataFrame(predicted_prices_arima).to_csv(f'{run_filepath}/ARIMA.csv')
    pd.DataFrame(predicted_prices_sarima).to_csv(f'{run_filepath}/SARIMA.csv')
    pd.DataFrame(predicted_prices_ensemble).to_csv(f'{run_filepath}/ENSEMBLE.csv')


    #------------------------------------------------------------------------------
    # Plot the test predictions
    #------------------------------------------------------------------------------
    if PLOT_INDIVIDUAL_RUN:
        # Plot the candlestick chart for the test data.
        plot_candlestick_chart(test_data, CANDLE_VIS_WINDOW,
             f"{settings['COMPANY']} Stock Price Chart", run_name, settings['TEST_NAME'])

        # Plot the boxplot chart for the predicted prices.
        plot_boxplot_chart(predicted_prices_lstm[settings['PRICE_VALUE']], BOX_VIS_WINDOW, f'{settings["COMPANY"]} Stock Price Boxplot Chart',
                           run_name=run_name, test_name=settings['TEST_NAME'], label_skips=28)

    if PLOT_ENSEMBLE:
        prediction_log = "Predictions: "
        predictions_all = []
        if settings['USE_LSTM']:
            predictions_all.append((predicted_prices_lstm, 'LSTM'))
            log_console_file(run_name, ' - Adding LSTM to predictions_all.')
            prediction_log += 'LSTM, '
        if settings['USE_ARIMA']:
            predictions_all.append((predicted_prices_arima, f'ARIMA\n{settings["arima_order"]}'))
            log_console_file(run_name, ' - Adding ARIMA to predictions_all.')
            prediction_log += 'ARIMA, '
        if settings['USE_SARIMA']:
            predictions_all.append((predicted_prices_sarima, f'SARIMA\n{settings["sarima_order"]}\nS:{settings["sarima_seasonal_order"]}'))
            log_console_file(run_name, ' - Adding SARIMA to predictions_all.')
            prediction_log += 'SARIMA, '
        if settings['USE_ENSEMBLE']:
            predictions_all.append((predicted_prices_ensemble, f'Ensemble\n({settings["ENSEMBLE_MODEL"]})'))
            log_console_file(run_name, ' - Adding Ensemble to predictions_all.')
            prediction_log += 'ENSEMBLE, '
        
        # Print all prediction labels to console
        log_console_file(run_name, f'------\nPlotting {prediction_log}')

        # Plot the test predictions including ensemble predictions
        plot_ensemble(test_data, actual_close_prices, predictions_all, price_value=settings['PRICE_VALUE'],
            run_name=run_name, test_name=settings['TEST_NAME'])

    return run_name

#------------------------------------------------------------------------------
# Run the model with the given settings
if RUN_MODEL:
    run_end_name = run_model(SETTINGS)
    log_console_file(run_end_name, "Run Complete.")
#------------------------------------------------------------------------------
if COMPARE_RUNS:
    if COMPARE_LOSS:
        # Default values will compare loss between LSTM models
        loss_plot_title = f"Training Loss History Comparison \n[{SETTINGS['DATA_SOURCE']}, {SETTINGS['COMPANY']}. [{SETTINGS['TRAIN_START']} - {SETTINGS['TRAIN_END']}]"
        compare_runs(TEST_NAME, COMPARISON_PARAMS_DICT['LSTM'], loss_plot_title)
    
    if COMPARE_PREDICTIONS:
        for prd in COMPARISON_PREDICTION_MODELS:
            for colkey, colval in COMPARISON_PREDICTION_COLUMN.items():
                prms = COMPARISON_PARAMS_DICT[prd].copy()
                # prms == list of strings | Needs to be merged with COMPARE_GLOBAL_PARAMS_LIST
                prms.extend(COMPARE_GLOBAL_PARAMS_LIST)

                plt_title = f"{prd} - {colkey} \n[{SETTINGS['DATA_SOURCE']}, {SETTINGS['COMPANY']}. [{SETTINGS['TRAIN_START']} - {SETTINGS['TRAIN_END']}]"
                compare_runs(TEST_NAME, prms, plt_title, prd, colkey, DISPLAY_PLOTS, SETTINGS['SEASONAL_PERIOD']//2, colval)
print("Done.")
#------------------------------------------------------------------------------