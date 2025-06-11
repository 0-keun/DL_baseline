import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, RNN, Dense, Dropout, Input
from keras import layers
import time
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy() # to use multiple GPUs


##############################
##           LSTM           ##
##############################

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
    

#############################
##           RNN           ##
#############################

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
    

##############################
##           FFNN           ##
##############################

def FFNN_model(feature_num, output_num):
    with strategy.scope():
        model = Sequential()
        model.add(Input(shape=(feature_num,)))   
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))  
        model.add(Dense(output_num, activation='linear'))  

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        # es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return model




###################################
##       Transformer Model       ##
###################################


def positional_encoding(position, d_model):
    import numpy as np
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Encoder Layer
def encoder_layer(d_model, num_heads, dff, dropout_rate=0.1):
    inputs = layers.Input(shape=(None, d_model))
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn = layers.Dropout(dropout_rate)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)
    ffn = layers.Dense(dff, activation='relu')(out1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return Model(inputs=inputs, outputs=out2)

# Decoder Layer
def decoder_layer(d_model, num_heads, dff, dropout_rate=0.1):
    inputs = layers.Input(shape=(None, d_model))      # target
    enc_outputs = layers.Input(shape=(None, d_model)) # encoder output

    # Masked multi-head attention (self-attention, future masking)
    attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs, attention_mask=None)
    attn1 = layers.Dropout(dropout_rate)(attn1)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn1)

    # Encoder-Decoder attention
    attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out1, enc_outputs)
    attn2 = layers.Dropout(dropout_rate)(attn2)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)

    ffn = layers.Dense(dff, activation='relu')(out2)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    out3 = layers.LayerNormalization(epsilon=1e-6)(out2 + ffn)

    return Model(inputs=[inputs, enc_outputs], outputs=out3)

# 전체 Transformer 모델
def transformer_model(
    input_vocab_size, target_vocab_size,
    num_layers=2, d_model=64, num_heads=4, dff=128,
    input_maxlen=100, target_maxlen=100, dropout_rate=0.1
):
    # ----- Encoder -----
    encoder_inputs = layers.Input(shape=(None,), name='encoder_inputs')
    enc_emb = layers.Embedding(input_vocab_size, d_model)(encoder_inputs)
    enc_emb += positional_encoding(input_maxlen, d_model)[:, :tf.shape(enc_emb)[1], :]
    enc_emb = layers.Dropout(dropout_rate)(enc_emb)

    x = enc_emb
    for _ in range(num_layers):
        x = encoder_layer(d_model, num_heads, dff, dropout_rate)(x)
    encoder_outputs = x

    # ----- Decoder -----
    decoder_inputs = layers.Input(shape=(None,), name='decoder_inputs')
    dec_emb = layers.Embedding(target_vocab_size, d_model)(decoder_inputs)
    dec_emb += positional_encoding(target_maxlen, d_model)[:, :tf.shape(dec_emb)[1], :]
    dec_emb = layers.Dropout(dropout_rate)(dec_emb)

    y = dec_emb
    for _ in range(num_layers):
        y = decoder_layer(d_model, num_heads, dff, dropout_rate)([y, encoder_outputs])
    decoder_outputs = y

    # ----- 최종 출력 -----
    final_outputs = layers.Dense(target_vocab_size, activation='softmax')(decoder_outputs)

    # 모델 정의
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=final_outputs)
    return model

    '''    
    model = transformer_model(
        input_vocab_size=1000,
        target_vocab_size=1000,
        num_layers=2,
        d_model=64,
        num_heads=4,
        dff=128,
        input_maxlen=100,
        target_maxlen=100
    )
    '''

