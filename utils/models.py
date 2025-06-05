import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, RNN, Dense, Dropout, Input
import time
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy() # to use multiple GPUs

def LSTM_model(hidden_state_num, class_num, time_steps, feature_num, layer_num):
    '''
    This model has same hidden_state_num for all hidden layers
    '''
    if class_num > 2:
        with strategy.scope(): 
            model = Sequential()
            model.add(Input(shape=(time_steps, feature_num)))

            for i in range(layer_num - 1):
                model.add(LSTM(hidden_state_num, return_sequences=True))
                model.add(Dropout(0.2))

            model.add(LSTM(hidden_state_num, return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(class_num, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        return model
    
    else:
        with strategy.scope(): 
            model = Sequential()
            model.add(Input(shape=(time_steps, feature_num)))

            for i in range(layer_num - 1):
                model.add(LSTM(hidden_state_num, return_sequences=True))
                model.add(Dropout(0.2))

            model.add(LSTM(hidden_state_num, return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(class_num, activation='sigmoid'))
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        return model
    
def RNN_model(hidden_state_num, class_num, time_steps, feature_num, layer_num):
    '''
    This model has same hidden_state_num for all hidden layers
    '''
    if class_num > 2:
        with strategy.scope(): 
            model = Sequential()
            model.add(Input(shape=(time_steps, feature_num)))

            for i in range(layer_num - 1):
                model.add(RNN(hidden_state_num, return_sequences=True))
                model.add(Dropout(0.2))

            model.add(RNN(hidden_state_num, return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(class_num, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        return model
    
    else:
        with strategy.scope(): 
            model = Sequential()
            model.add(Input(shape=(time_steps, feature_num)))

            for i in range(layer_num - 1):
                model.add(RNN(hidden_state_num, return_sequences=True))
                model.add(Dropout(0.2))

            model.add(RNN(hidden_state_num, return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(class_num, activation='sigmoid'))
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        return model