# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance
from collections import deque # Added - Weekly Report 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf

import os # Added - Weekly Report 2
from sklearn.model_selection import train_test_split # Added - Weekly Report 2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer


DATA_SOURCE = "yahoo"
COMPANY = "TSLA"

# start = '2012-01-01', end='2017-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

PREDICTION_DAYS_PAST = 60 # Modified variable name for clarity
### Newly Added - Weekly Report 2
PREDICT_DAYS_AHEAD = 1 
TEST_SIZE = 0.2
SHUFFLE_DATASET = True
SPLIT_BY_DATE = True
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
PRICE_VALUE = "Close"
LOCAL_SAVE=True
LOCAL_LOAD=True
LOCAL_PATH='./data'
SCALE_FEATURES=True
EPOCHS=25
BATCH_SIZE=32

# Pinched from P1
def shuffle_in_unison(a, b):
    # shuffle both arrays using the same random seed
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------

def load_data_and_process(company, data_source, start_date, end_date, predict_days,
                          scale=True, split_by_date=True, test_size=0.5, shuffle=True,
                          feature_columns=FEATURE_COLUMNS, prediction_stepsize=1,
                          load_local=True, save_local=True, local_path=LOCAL_PATH):
    """
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

    Returns:
        result: a dict containing the following:
            df (pd.DataFrame): the original dataframe
            column_scaler (dict): the MinMaxScaler instances for each feature
            last_sequence (np.array): the last sequence of the training data
            x_train (np.array): the training data
            y_train (np.array): the training labels
            x_test (np.array): the testing data
            y_test (np.array): the testing labels
            test_df (pd.DataFrame): the testing dataframe           
    """
    # Check if data has been prepared before.
    if load_local and os.path.exists(os.path.join(local_path, company + '.csv')):
        # Load the saved data as a dataframe from local storage
        data = pd.read_csv(os.path.join(local_path, company + '.csv'))
    else:
        # Download the data as a dataframe
        if isinstance(company, str):
            if data_source == "yahoo":
                data = yf.download(company, start=start_date, end=end_date, progress=False)
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

        # Save the data into a directory
        if save_local:
            # Save the data to local storage as a dataframe
            data.to_csv(os.path.join(local_path, company + '.csv'))

    result = {} # Storing result in a dict for convenience
    result['df'] = data.copy() # include original dataframe copy too

    # Make sure that all feature_columns exist in the dataframe
    for col in feature_columns:
        # Will throw an assert error if the column does not exist
        assert col in data.columns, f"'{col}' does not exist in the dataframe."

    # Add date as a column
    if "date" not in data.columns:
        data["date"] = data.index

    if scale:
        column_scaler = {}
        # Scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = MinMaxScaler()
            # Using np.expand_dims to reshape the data from (x,) to (x,1)
            # in order to fit MinMaxScaler, which requires a 2D array.
            # (much more elegant than .reshape(-1,1), more akin to pytorch unsqueeze())
            # https://note.nkmk.me/en/python-numpy-newaxis/
            data[column] = scaler.fit_transform(np.expand_dims(data[column].values, axis=1))
            column_scaler[column] = scaler

        # Add the MinMaxScaler instances to the result returned, this will allow us to
        # convert back to actual prices later.
        result["column_scaler"] = column_scaler


    # Add the target column (label) by shifting by `prediction_stepsize`
    # Currently still using Close price (the value of PRICE_VALUE) for now.
    data['future'] = data[PRICE_VALUE].shift(-prediction_stepsize)
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
    for entry, target in zip(data[feature_columns + ["date"]].values, data["future"].values):
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
  
        if shuffle: # training and testing datasets are shuffled.
            # Using function shuffle_in_unison() borrowed from P1
            shuffle_in_unison(result["x_train"], result["y_train"])
            shuffle_in_unison(result["x_test"], result["y_test"])
    else:
        # If split_by_date is False, dataset is split randomly with train_test_split()
        result["x_train"], result["x_test"], result["y_train"], result["y_test"] = \
            train_test_split(train_x, train_y, test_size=test_size, shuffle=shuffle)

    # result["x_train"] and result["x_test"] are three-dimensional arrays, where the second 
    # dimension contains the sequences and the third dimension holds the features and the date.
    dates = result["x_test"][:, -1, -1]
    # Since the third dimension of result["x_test"] holds the date information for each sequence,
    # we can extract the date of each test sequence with [:, -1, -1] and store it in an array 'dates'.

    result["test_df"] = result["df"].loc[dates]
    # result["df"] holds the original DataFrame containing the entire dataset.
    # we can use the extracted dates array to index and select rows from the original
    # DataFrame, effectively creating a new DataFrame containing only the rows
    # corresponding to the testing data dates. (resulting DataFrame then stored in result["test_df"].)

    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # Here we remove duplicated rows from the testing DataFrame while retaining the
    # first occurrence of each duplicated date.
    # The resulting DataFrame without duplicated dates is stored back in result["test_df"].

    # Lastly, we modify the training and testing data arrays stored in result to ensure
    # that they contain only the feature columns and not the date information.
    result["x_train"] = result["x_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["x_test"] = result["x_test"][:, :, :len(feature_columns)].astype(np.float32)
    # As with before, we can use slice notation with [:, :, :len(feature_columns)] to effectively
    # remove the date information from each array, leaving only the feature columns.

    return result # Return the result dict containing the processed data

loaded_data = load_data_and_process(company=COMPANY,
                                    data_source=DATA_SOURCE,
                                    start_date=TRAIN_START,
                                    end_date=TRAIN_END,
                                    predict_days=PREDICTION_DAYS_PAST,
                                    scale=SCALE_FEATURES,
                                    split_by_date=SPLIT_BY_DATE,
                                    test_size=TEST_SIZE,
                                    shuffle=SHUFFLE_DATASET,
                                    feature_columns=FEATURE_COLUMNS,
                                    prediction_stepsize=PREDICT_DAYS_AHEAD,
                                    load_local=LOCAL_LOAD,
                                    save_local=LOCAL_SAVE,
                                    local_path=LOCAL_PATH)

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(PREDICTION_DAYS_PAST, len(FEATURE_COLUMNS))))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(loaded_data['x_train'], loaded_data['y_train'], epochs=EPOCHS, batch_size=BATCH_SIZE)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'
test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

'''
test_data = load_data_and_process(company=COMPANY,
                                    data_source=DATA_SOURCE,
                                    start_date=TEST_START,
                                    end_date=TEST_END,
                                    predict_days=PREDICTION_DAYS_PAST,
                                    scale=SCALE_FEATURES,
                                    split_by_date=SPLIT_BY_DATE,
                                    test_size=TEST_SIZE,
                                    shuffle=SHUFFLE_DATASET,
                                    feature_columns=FEATURE_COLUMNS,
                                    prediction_stepsize=PREDICT_DAYS_AHEAD,
                                    load_local=False,
                                    save_local=False)
'''
# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((loaded_data['df'][PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS_PAST:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line
model_inputs = loaded_data["column_scaler"][PRICE_VALUE].transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS_PAST, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS_PAST:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = loaded_data["column_scaler"][PRICE_VALUE].inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS_PAST:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = loaded_data["column_scaler"][PRICE_VALUE].inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??