import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import dates
from datetime import datetime as dt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

import numpy as np
import datetime
import glob
import time



class DispenserBehavior():
    def __init__(self, data, resample_interval = '30T', scaler = 'standardize'):
        self.data = data.copy()
        self.resample_interval = resample_interval

        if scaler == 'standardize':
            self.scaler = StandardScaler()
        elif scaler == 'normalize':
            self.scaler = MinMaxScaler()

        # Calculate the total usage of water dispenser data
        self.data.Usage_CC = self.__calculate_usage(self.data.Usage_CC, self.data.Usage_L, self.data.Usage_MT)
        self.data.drop(columns=['Usage_L', 'Usage_MT'], inplace=True)
        self.data.UploadTime = pd.to_datetime(self.data.UploadTime, format='%Y-%m-%d %H:%M:%S')
        self.data.drop_duplicates(subset=['UploadTime'], inplace=True)
        self.data.set_index('UploadTime', inplace=True)

        # Resample the data into certain interval
        features = self.data.drop(columns=['HotTemp', 'WarmTemp', 'ColdTemp'])
        features = features.resample(self.resample_interval).sum()
        features.reset_index(inplace=True)

        tanks_temp = self.data[['HotTemp', 'WarmTemp', 'ColdTemp']].resample(self.resample_interval).last()
        tanks_temp = tanks_temp.fillna(method = 'ffill')
        tanks_temp = tanks_temp.astype('int64')
        tanks_temp.reset_index(inplace=True, drop = True)

        self.data_resampled = pd.concat([features,tanks_temp], axis = 1)
        
        # Scaling the data with normalization / standardization
        df_numpy = self.data_resampled.drop(columns = ['UploadTime']).to_numpy()
        self.data_rescaled = self.scaler.fit_transform(df_numpy)
        
        
    def __calculate_usage(self, cc, l, mt):
        total_usage = lambda a, b, c : a + (b*1000) + (c*1000000)
        usage = []

        for i in dispenser_data.index:
            if i == 0:
                usage.append(0)
                prev = total_usage(cc[i], l[i], mt[i])
                continue

            temp = total_usage(cc[i], l[i], mt[i])
            usage.append(temp - prev)
            prev = temp 
        
        return usage
    
    """Method for windowing the data with some time steps"""
    
    def data_windowing(self, data, n_steps, labels = None):
        temp_X, temp_y = list(), list()
        
        if labels.all() != None:
            for i in range(len(data)):
                end_ix = i + n_steps

                if end_ix > len(data)-1:
                    break

                seq_x, seq_y = data[i:end_ix, :], labels[end_ix]
                temp_X.append(seq_x)
                temp_y.append(seq_y)
        else:
            for i in range(len(data)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(data)-1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
                temp_X.append(seq_x)
                temp_y.append(seq_y)
            
        self.X = np.array(temp_X)
        self.y = np.array(temp_y)
        
        print('Shape of input features (X variable): ', self.X.shape)
        print('Shape of labels (y variable): ', self.y.shape)
    
    
    """Method for splitting the dataset into training, validation, testing or training and testing only"""
    
    def data_splitting(self, train_size, val_size = 0):
        if train_size + val_size > 1.0:
            return " The sum of train, test, and validation set should be equals to 1.0!"
        
        if val_size != 0:        
            train_index = int(self.X.shape[0]*train_size) 
            val_index = int(self.X.shape[0]*val_size)
            self.train_index = train_index        
            self.val_index = val_index
            
            self.X_train = self.X[:train_index]
            self.X_val = self.X[train_index:train_index+val_index]
            self.X_test = self.X[train_index+val_index:]

            self.y_train = self.y[:train_index]
            self.y_val = self.y[train_index:train_index+val_index]
            self.y_test = self.y[train_index+val_index:]
            
            print('X_val:', self.X_val.shape, '| y_val:', self.y_val.shape)
            
        else:
            train_index = int(self.X.shape[0]*train_size)         
            self.train_index = train_index
            self.val_index = None
            
            self.X_train = self.X[:train_index]
            self.X_val = None
            self.X_test = self.X[train_index:]

            self.y_train = self.y[:train_index]
            self.y_val = None
            self.y_test = self.y[train_index:]
            
        print('X_train:',self.X_train.shape, '| y_train:', self.y_train.shape)
        print('X_test:', self.X_test.shape, '| y_test:', self.y_test.shape)

        
    """Method for training the dataset with some hyperparameter"""
    
    def train(self, epochs, output, lstm_units = 15, dropout = 0.4, dense_units = 30, l2_decay = 0.1, loss_func = 'mean_absolute_error', patience = 10, verbose = 1):
        
        n_steps = self.X.shape[1]
        n_features = self.X.shape[2]
        
        self.LSTM_model = tf.keras.Sequential()
        self.LSTM_model.add(LSTM(units = lstm_units, activation='relu', input_shape=(n_steps, n_features)))    
        self.LSTM_model.add(Dropout(dropout))
        self.LSTM_model.add(Dense(dense_units, kernel_regularizer = regularizers.l2(l2_decay)))
        self.LSTM_model.add(Dense(output))
        self.LSTM_model.compile(optimizer = 'adam', loss = loss_func)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = patience)

        start_time = time.time()
        if self.X_val == None:
            self.history = self.LSTM_model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), callbacks = [es], epochs = epochs, verbose = verbose)
        else:
            self.history = self.LSTM_model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), callbacks = [es], epochs= epochs, verbose = verbose)
        self.training_time = time.time() - start_time
        
        print('\n\n\n============================\nTraining and testing evaluation: \n')
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'])
        plt.show()
        
        self.LSTM_model.evaluate(self.X_test, self.y_test)
        print('Training time %.3f seconds' % self.training_time)
    
    """ Method for predicting and compare the prediction result with labels"""
    
    
    def predict_and_compare(self, test_input, test_labels):
        
        yhat = self.LSTM_model.predict(model.X_test)
        yhat = self.scaler.inverse_transform(yhat)

        if self.y.shape[1] == 1:
            self.predicted_values = list(yhat.flatten().astype('int64'))
        else:
            self.predicted_values = yhat
        
        self.__plot_prediction(yhat, test_labels)
    
    """ Method for plotting the prediction result """"
    
    def __plot_prediction(self, yhat, y_val):
    
        fig, ax = plt.subplots(3, 1, figsize=(18, 15))
        fig.suptitle('Water Usage Prediction')

        if self.val_index != None:
            time_steps = list(self.data_resampled.iloc[self.train_index + self.val_index + self.X.shape[1]:].UploadTime)
        else:
            time_steps = list(self.data_resampled.iloc[(self.train_index + self.X.shape[1]):].UploadTime)

        index = int(0.33*len(time_steps))

        predicted_marker = mlines.Line2D([], [], marker='.', linestyle='None',
                              markersize=10, label='Actual')
        actual_marker = mlines.Line2D([], [], color='orange', marker='X', linestyle='None',
                              markersize=10, label='Predicted')

        for i in range(0, 3):
            ax[i].set_xlabel('Time Steps [30m]\n')
            ax[i].set_ylabel('Water Usage')
            ax[i].legend(handles = [actual_marker, predicted_marker])

            if i == 2:
                ax[i].plot(time_steps[i*index:], y_val[i*index:], linestyle = '-', marker = '.', markersize = 10)
                ax[i].plot(time_steps[i*index:], yhat[i*index:], color='orange', linestyle = '', marker = 'X', markersize = 8)

            else:
                ax[i].plot(time_steps[i*index:(i+1)*index], y_val[i*index:(i+1)*index], linestyle = '-', marker = '.', markersize = 10)
                ax[i].plot(time_steps[i*index:(i+1)*index], yhat[i*index:(i+1)*index], color='orange', linestyle = '', marker = 'X', markersize = 8)

