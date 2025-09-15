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
    
def FFNN_model(feature_num, output_num):
    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(feature_num,)))   
        model.add(Dense(256, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))  
        model.add(Dense(output_num, activation='linear'))  

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        # es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return model

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import gpytorch
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualFFBlock(nn.Module):
    """
    Fully-connected residual block.
    in_dim -> hidden_dim -> out_dim, 필요 시 residual projection으로 차원 맞춤
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout: float = 0.0):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(in_dim, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, out_dim))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # in_dim != out_dim이면 projection으로 residual 차원 정렬
        if in_dim != out_dim:
            self.proj = spectral_norm(nn.Linear(in_dim, out_dim))
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + self.proj(residual)
        return self.act(out)


class FeatureExtractorRes(nn.Module):
    """
    SN 기반 FFNN + Residual 연결 (입력 차원 → hidden 승격 후 hidden 영역에서 잔차)
    기존 FeatureExtractor와 동일한 시그니처 유지: (input_dim, hidden_dim=512, output_dim=64)
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=64, dropout: float = 0.0):
        super().__init__()
        print("FFNN with Spectral Normalization + Residual connections is adopted.")

        # 1) input -> hidden (lift)
        self.fc_in = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU()
        )

        # 2) hidden 구간에서만 residual block 적용
        self.block1 = ResidualFFBlock(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.block2 = ResidualFFBlock(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # 3) hidden -> output
        self.output_layer = spectral_norm(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.fc_in(x)      # (B, hidden_dim)
        x = self.block1(x)     # (B, hidden_dim)
        x = self.block2(x)     # (B, hidden_dim)
        return self.output_layer(x)  # (B, output_dim)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, output_dim))
        )

    def forward(self, x):
        return self.net(x)
    
class FeatureExtractorWithoutSN(nn.Module):
    def __init__(self, input_dim, hidden_dim=255, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DKLGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, feature_extractor, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

