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
from cgi import test
from collections import deque
from multiprocessing.pool import RUN


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import datetime as dt
import tensorflow as tf
import yfinance as yf
import mplfinance as mpf

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU

DATA_SOURCE = "yahoo"
COMPANY = "TSLA"

# start = '2012-01-01', end='2017-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

TEST_START = '2020-01-02'
TEST_END = '2022-12-31'

PREDICTION_DAYS_PAST = 60 # number of days to look back to base the prediction | Default: 60
PREDICT_DAYS_AHEAD = 1
TEST_SIZE = 0.2
SHUFFLE_DATASET = True
SPLIT_BY_DATE = True
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PRICE_VALUE = "Close"
LOCAL_SAVE = True
LOCAL_LOAD = True
LOCAL_PATH = './data'
SCALE_FEATURES = True

# Model parameters
CELL = LSTM # the type of cell to use for the DL network | LSTM, GRU | Default: LSTM
EPOCHS = 25 # number of epochs (repetitions of training) to train for | Default: 25
BATCH_SIZE = 32 # number of samples per update | Default: 32
LOSS = "mean_absolute_error" # loss function used to calculate prediction error | 'mean_absolute_error', 'huber_loss', 'mean_squared_error' | Default: "mean_absolute_error"
OPTIMIZER = 'adam' # which optimizer function to use | 'adam', 'rmsprop', 'sgd' | Default: "adam"
NUM_LAYERS = 2 # Default: 2
DROPOUT = 0.2 # dropout defines the fraction of deactivated neurons, eg; 0.2 == 20%.
UNITS = 50 # the number of units/nodes in each layer | Default: 50

# Get variable values for the plot and filename
RUN_NAME = "mean_absolute_error"
EPOCHS_NAME = str(EPOCHS)
CELL_NAME = str(CELL).rsplit(sep='.', maxsplit=1)[-1].rsplit(sep="'", maxsplit=1)[0]
BATCH_NAME = str(BATCH_SIZE)
LAYERS_NAME = str(NUM_LAYERS)
UNITS_NAME = str(UNITS)
DROP_NAME = str(DROPOUT)
OPTIM_NAME = str(OPTIMIZER)
LOSS_NAME = str(LOSS)

# Save parameter settings
settings = {
'epochs' : EPOCHS_NAME,
'batch_size' : BATCH_NAME,
'layers' : LAYERS_NAME,
'units' : UNITS_NAME,
'dropout' : DROP_NAME,
'cell' : CELL_NAME,
'optimizer' : OPTIM_NAME,
'loss_function' : LOSS_NAME,
'DATA_SOURCE' : DATA_SOURCE,
'COMPANY' : COMPANY,
'TRAIN_START' : TRAIN_START,
'TRAIN_END' : TRAIN_END,
'TEST_START' : TEST_START,
'TEST_END' : TEST_END,
'PREDICTION_DAYS_PAST' : PREDICTION_DAYS_PAST,
'PREDICT_DAYS_AHEAD' : PREDICT_DAYS_AHEAD,
'TEST_SIZE' : TEST_SIZE,
'SHUFFLE_DATASET' : SHUFFLE_DATASET,
'SPLIT_BY_DATE' : SPLIT_BY_DATE,
'FEATURE_COLUMNS' : str(FEATURE_COLUMNS),
'PRICE_VALUE' : PRICE_VALUE,
'LOCAL_SAVE' : LOCAL_SAVE,
'LOCAL_LOAD' : LOCAL_LOAD,
'LOCAL_PATH' : LOCAL_PATH,
'SCALE_FEATURES' : SCALE_FEATURES
}

#------------------------------------------------------------------------------
# Helper Functions

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

def load_data_and_process(company, data_source, predict_days,
                          train_start_date, train_end_date, 
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
    # -------- Load the data --------
    # Check if data has been prepared before.
    if load_local and os.path.exists(os.path.join(local_path, company + '.csv')):
        # Load the saved data as a dataframe from local storage
        data = pd.read_csv(os.path.join(local_path, company + '.csv'), parse_dates=True, index_col=0)
    else:
        # Download the data as a dataframe
        if isinstance(company, str):
            if data_source == "yahoo":
                data = yf.download(company, start=train_start_date, end=train_end_date, progress=False, parse_dates=True, index_col=0)
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
            # Save the data to local storage as a dataframe
            data.to_csv(os.path.join(local_path, company + '.csv'))

    result = {} # Storing result in a dict for convenience
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

    # Lastly, we modify the training and testing data arrays stored in result to ensure
    # that they contain only the feature columns and not the date information.
    result["x_train"] = result["x_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["x_test"] = result["x_test"][:, :, :len(feature_columns)].astype(np.float32)
    # As with before, we can use slice notation with [:, :, :len(feature_columns)] to effectively
    # remove the date information from each array, leaving only the feature columns.

    return result # Return the result dict containing the processed data

loaded_data = load_data_and_process(company=COMPANY,
                                    data_source=DATA_SOURCE,                                   
                                    predict_days=PREDICTION_DAYS_PAST,
                                    train_start_date=TRAIN_START,
                                    train_end_date=TRAIN_END,
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
# Change the model to increase accuracy?
#------------------------------------------------------------------------------

# Create the function to build the model
def create_model(n_feats, seq_len=60, units=50, cell=LSTM, n_lays=2, dropout=0.2,
                loss="mean_absolute_error", optim="adam"):
    """
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
    def __init__(self):
        super().__init__()
        self.loss_history = {'loss': {}}

    def on_epoch_end(self, epoch, logs=None):
        # We append the loss metric to the history dict
        self.loss_history["loss"][epoch] = logs["loss"]

# ------------------------ Training the model ------------------------
RUN_MODEL = True

if RUN_MODEL:
    history_callback = LossCallback()

    # Call the function to create our model
    model = create_model(
        n_feats = len(FEATURE_COLUMNS), # number of features to use to feed into the model
        seq_len = PREDICTION_DAYS_PAST, # number of days to look back to base the prediction
        units   = UNITS, # number of units in each layer
        cell    = CELL, # the type of cell to use
        n_lays  = NUM_LAYERS, # the number of layers
        # Dropout is a regularization technique to help prevent overfitting by randomly deactivating
        # neurons during training to encourage the network to learn more robust and generalizable
        # features, since the network is penalized for over-relying on any particular neuron.
        # Dropout variable is the fraction of the neurons that are deactivated, so 0.2 == 20%.
        dropout = DROPOUT, # the dropout rate
        loss    = LOSS, # the loss function
        optim   = OPTIMIZER) # the optimizer function

    # Get run start time:
    time_started = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # (x_train, y_train)
    # DL Networks: LSTM, RNN, GRU, CNN
    # epochs
    # batch_size
        # The optimizer and loss are two important parameters when building an
        # ANN model. Choosing a different optimizer/loss can affect the prediction
        # quality significantly. You should try other settings to learn; e.g.
    # optimizer='rmsprop'/'sgd'/'adadelta'/...
    # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

    model.fit(loaded_data['x_train'], loaded_data['y_train'], epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[history_callback])

    # Get run end time:
    time_ended = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f'{COMPANY}_-_{time_ended}'

    # Access loss_history dict after training (and print values)
    loss_history = history_callback.loss_history
    print(loss_history['loss'])

    # Can ignore the first epoch since it's usually a lot higher than the rest

    ### Plot the loss history
    # Extract the loss values and epoch numbers (except first epoch)
    epochs = list(loss_history["loss"].keys())[1:]
    loss_values = list(loss_history["loss"].values())[1:]

    # Make directory if it doesn't exist
    if not os.path.exists(os.path.join(f'./logs/{RUN_NAME}/')):
        os.makedirs(os.path.join(f'./logs/{RUN_NAME}/'))
    if not os.path.exists(os.path.join(f'./logs/{RUN_NAME}/')):
        os.makedirs(os.path.join(f'./logs/{RUN_NAME}/'))
    if not os.path.exists(os.path.join(f'./logs/{RUN_NAME}/', run_name)):
        os.makedirs(os.path.join(f'./logs/{RUN_NAME}/', run_name))

    ## Display loss plot
    # Create a plot
    plt.figure(figsize=(12, 6))
    fig = plt.get_current_fig_manager()
    fig.set_window_title(f'Model: {COMPANY} Stock Price Prediction | ./{run_name}/loss_plot.jpg')
    plt.plot(epochs, loss_values, marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.title(f'Model: {COMPANY} Stock Price Prediction | \
    Total Epochs : {EPOCHS_NAME}\n\
    Batch Size: {BATCH_NAME} | \
    Layer Count: {LAYERS_NAME} |\
    Units (per Layer): {UNITS_NAME} |\
    Dropout: {DROP_NAME}\n\
    Cell: {CELL_NAME} |\
    Optimizer: {OPTIM_NAME} |\
    Loss Function: {LOSS_NAME}')
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss'])

    # Save before display since display will clear the plot
    plt.savefig(f'./logs/{RUN_NAME}/{run_name}/loss_plot.jpg', dpi=300, bbox_inches='tight')
    plt.show()


    ## Save loss history
    loss_df = pd.DataFrame(loss_history)
    # Save the dataframe to CSV
    loss_df.to_csv(f'./logs/{RUN_NAME}/{run_name}/loss.csv')

    ## Save model
    model.save(f'./logs/{RUN_NAME}/{run_name}/model.h5')

    # Save settings to CSV
    settings['run_name'] = run_name
    settings['time_started'] = time_started
    settings['time_ended'] = time_ended

    if not os.path.exists(f'./logs/{RUN_NAME}/training_runs.csv'):
        settings_df = pd.DataFrame(settings, index=[time_ended])
        settings_df.to_csv(f'./logs/{RUN_NAME}/training_runs.csv', index_label='timestamp')
    else:
        settings_df = pd.DataFrame(settings, index=[time_ended])
        settings_df.to_csv(f'./logs/{RUN_NAME}/training_runs.csv', index_label='timestamp', mode='a', header=False)



    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------
    # Load the test data
    test_data = loaded_data['test_df']#.drop(columns=['Date'])

    actual_close_prices = loaded_data['test_df'][PRICE_VALUE].values

    # We need to do the following because to predict the closing price of the fisrt
    # PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period
    total_dataset = pd.concat((loaded_data['df'][FEATURE_COLUMNS], test_data[FEATURE_COLUMNS]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS_PAST:].values

    # Normalize the data within each column using the column scaler value
    for i, feat in enumerate(FEATURE_COLUMNS):
        print(i, feat)
        print(model_inputs.shape)
        model_inputs[:, i] = loaded_data["column_scaler"][feat].transform(model_inputs[:, i].reshape(-1, 1)).reshape(-1)
        #------------------------------------------------------------------------------
        # Make predictions on test data
        #------------------------------------------------------------------------------
    x_test = []
    for x in range(PREDICTION_DAYS_PAST, len(model_inputs)):
        x_test.append(model_inputs[x-PREDICTION_DAYS_PAST:x])


    x_test = np.array(x_test)
    x_train = np.reshape(x_test, (-1, x_test.shape[1], len(FEATURE_COLUMNS)))
    # Reshaping the data into the format that the model expects:
    # (number of samples, number of time steps, number of features per sample)

    predicted_prices = model.predict(x_test).reshape(-1)
    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 

    #------------------------------------------------------------------------------
    # Predict next day
    #------------------------------------------------------------------------------

    real_data = []
    for x in range(PREDICTION_DAYS_PAST, len(model_inputs)):
        real_data.append(model_inputs[x-PREDICTION_DAYS_PAST:x])

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (-1, real_data.shape[1], len(FEATURE_COLUMNS)))

    prediction = model.predict(real_data)
    predicted_prices = {}
    #actual_prices = {}
    # loaded_data["column_scaler"][price_val].inverse_transform(prediction).reshape(-1)
    for feat in FEATURE_COLUMNS:
        predicted_prices[feat] = loaded_data["column_scaler"][feat].inverse_transform(prediction).reshape(-1)
        #actual_prices = loaded_data['df'][feat].values

    # include deviation from actual price in prediction dict
    predicted_prices['close_price_deviation'] = predicted_prices[PRICE_VALUE] - actual_close_prices

    print(f"Prediction: {predicted_prices[PRICE_VALUE]}")

    ## Save predictions to CSV
    pred_df = pd.DataFrame(predicted_prices)
    pred_df.to_csv(f'./logs/{RUN_NAME}/{run_name}/predictions.csv')

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

#------------------------------------------------------------------------------
# Plot the test predictions
#------------------------------------------------------------------------------
def plot_candlestick_chart(data, window_size=1, title="Candlestick Chart"):
    """
    Plot a candlestick chart for stock market financial data.

    Args:
        data (pd.DataFrame): The financial data with columns 'Open', 'High', 'Low', 'Close', and 'Date'.
        candlestick_window (int): The number of consecutive trading days to represent as a single candlestick.
        title (str): The title of the chart.

    Returns:
        None (displays the chart).
    """
    # Resample the data to create candlesticks representing 'candlestick_window' days.
    # Resample works by grouping the data by a certain time period and then aggregating
    # the data within each time period. In this case, we group by 'candlestick_window' days
    # and aggregate using the first, max, min, and last prices for each group.
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
             ylabel='Price', ylabel_lower='Date', show_nontrading=True)

def plot_boxplot_chart(data, window_size=2, title="Boxplot Chart", label_skips=14):
    """
    Plot a boxplot chart for stock market financial data with a moving window.

    Args:
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
    plot_data = [data.reshape(-1)[i:i+window_size] for i in range(num_boxplots)]

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
    for nth, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if nth % label_skips != 0:
            label.set_visible(False)
    plt.show()

#------------------------------------------------------------------------------
CANDLE_VIS_WINDOW = 10
BOX_VIS_WINDOW = 90

PLOT_INDIVIDUAL_RUN = False
COMPARE_RUNS = True

if PLOT_INDIVIDUAL_RUN:
    # Plot the candlestick chart for the test data.
    plot_candlestick_chart(test_data, window_size=CANDLE_VIS_WINDOW, title=f'{COMPANY} Stock Price Chart')
    # Plot the boxplot chart for the predicted prices.
    plot_boxplot_chart(prediction, window_size=BOX_VIS_WINDOW, title=f'{COMPANY} Stock Price Boxplot Chart', label_skips=28)
    
if COMPARE_RUNS:
    # Compare the loss history of different runs with a subplot for each run
    runs = pd.read_csv(f'./logs/{RUN_NAME}/training_runs.csv', index_col=0)

    plt.figure(figsize=(12, 6))
    fig = plt.get_current_fig_manager()
    style_cur = col_cur = 0
    style_list = ['-', '--', '-.', ':']
    col_list = ['r', 'g', 'b', 'k']
    # Get list of parameter values that are fixed/identical among all training runs
    params = ['epochs', 'batch_size', 'layers', 'units', 'dropout', 'cell',
              'optimizer', 'loss_function', 'PREDICTION_DAYS_PAST', 'PREDICT_DAYS_AHEAD',
              'TEST_SIZE', 'SHUFFLE_DATASET', 'SPLIT_BY_DATE', 'FEATURE_COLUMNS', 'SCALE_FEATURES']

    identical = (runs[params].nunique() == 1)
    diff = identical[identical is False].index.tolist()
    identical = identical[identical is True].index.tolist()
    
    for i, row in runs.iterrows():
        loss_history_path = f'./logs/{RUN_NAME}/{row["run_name"]}/loss.csv'
        loss_history_df = pd.read_csv(loss_history_path, index_col=0)
        # Can ignore the first epoch since it's usually a lot higher than the rest
        loss_history_df = loss_history_df.iloc[1:]

        # change color for each run, only change style when color has been used
        run_differences = row[diff].to_dict()
        
        plt.plot(loss_history_df['loss'], label=f'{row["run_name"]} | {run_differences}', linestyle=style_list[style_cur%4], color=col_list[col_cur%4])
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

    # Print or use the resulting string as needed
    print(label)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title(f'Run Loss History Comparison | {DATA_SOURCE}, {COMPANY}. [{TRAIN_START} - {TRAIN_END}]\nSettings: [{label}]')
    plt.legend()
    plt.savefig(f'./logs/{RUN_NAME}/run_comparison.jpg', dpi=300, bbox_inches='tight')
    plt.show()