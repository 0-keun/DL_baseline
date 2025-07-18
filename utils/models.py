import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, RNN, Dense, Dropout, Input
from keras import layers
import time
import tensorflow as tf
import os
from utils.utils import name_to_dir

strategy = tf.distribute.MirroredStrategy() # to use multiple GPUs

# 사용자 정의 콜백 클래스
class SaveEveryNEpoch(tf.keras.callbacks.Callback):
    def __init__(self, save_path, interval=100):
        super(SaveEveryNEpoch, self).__init__()
        self.save_path = save_path
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            model_path = os.path.join(self.save_path)
            self.model.save(model_path)
            print(f'\n✅ 모델이 {epoch+1}번째 epoch 후 저장되었습니다: {model_path}')

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
    

# ──────────────── Adversarial Training용 모델 ────────────────
import tensorflow as tf

class AdversarialLSTM(tf.keras.Model):
    def __init__(self, base_model,
                 attack_method='bim',  # 'fgsm', 'bim', or 'pgd'
                 eps=0.01,
                 alpha=0.005,
                 iters=3):
        """
        attack_method: 'fgsm' | 'bim' | 'pgd'
        eps          : L∞ 볼 반경
        alpha        : 한 스텝당 perturb 크기 (bim/pgd)
        iters        : 반복 횟수 (bim/pgd)
        """
        super().__init__()
        self.base_model = base_model
        self.attack_method = attack_method
        self.eps = eps
        self.alpha = alpha
        self.iters = iters

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def _fgsm(self, x, y):
        # 단일-step FGSM
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self(x, training=True)
            loss = self.compiled_loss(y, preds)
        grad = tape.gradient(loss, x)
        x_adv = x + self.eps * tf.sign(grad)
        return tf.clip_by_value(x_adv, 0.0, 1.0)

    def _bim(self, x, y):
        # I‑FGSM: BIM without random start
        x_adv = tf.identity(x)
        for _ in range(self.iters):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                preds = self(x_adv, training=True)
                loss = self.compiled_loss(y, preds)
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv + self.alpha * tf.sign(grad)
            # 원본 x 주변 eps 이내로 클리핑
            x_adv = tf.clip_by_value(x_adv, x - self.eps, x + self.eps)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv

    def _pgd(self, x, y):
        # PGD: random start + BIM
        # 1) 랜덤 초기화
        x_adv = x + tf.random.uniform(tf.shape(x), 
                                      minval=-self.eps, maxval=self.eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        # 2) 반복적 FGSM + 투사
        for _ in range(self.iters):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                preds = self(x_adv, training=True)
                loss = self.compiled_loss(y, preds)
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv + self.alpha * tf.sign(grad)
            # x 주변 eps 이내, 그리고 [0,1] 범위로 클리핑
            x_adv = tf.clip_by_value(x_adv, x - self.eps, x + self.eps)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv

    def train_step(self, data):
        x, y = data

        # 1) 공격 예시 생성
        if self.attack_method == 'fgsm':
            x_adv = self._fgsm(x, y)
        elif self.attack_method == 'bim':
            x_adv = self._bim(x, y)
        elif self.attack_method == 'pgd':
            x_adv = self._pgd(x, y)
        else:
            raise ValueError(f"Unknown attack method: {self.attack_method}")

        # 2) 원본 + 적대 예시 합치기
        x_comb = tf.concat([x, x_adv], axis=0)
        y_comb = tf.concat([y, y], axis=0)

        # 3) 학습
        with tf.GradientTape() as tape:
            preds = self(x_comb, training=True)
            loss = self.compiled_loss(y_comb, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 4) metric 업데이트
        self.compiled_metrics.update_state(y_comb, preds)
        return {m.name: m.result() for m in self.metrics}



# ──────────────── 기존 LSTM_model 함수 수정 ────────────────
def LSTM_model_ADV(hidden_state_num, class_num, time_steps, feature_num, layer_num, eps=0.01):
    """
    AdversarialLSTM으로 wrapping된 모델을 반환합니다.
    eps: FGSM perturbation 크기
    """
    with strategy.scope():
        # 1) 기본 Sequential 아키텍처 생성
        base = Sequential()
        base.add(Input(shape=(time_steps, feature_num)))
        for _ in range(layer_num - 1):
            base.add(LSTM(hidden_state_num, return_sequences=True))
            base.add(Dropout(0.2))
        base.add(LSTM(hidden_state_num))
        base.add(Dropout(0.2))

        # 클래스 수에 따라 마지막 activation/ loss 설정
        if class_num > 2:
            base.add(Dense(class_num, activation='softmax'))
            loss_fn = 'categorical_crossentropy'
        else:
            base.add(Dense(class_num, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'

        # 2) Adversarial 원본 모델 감싸기
        model = AdversarialLSTM(base, attack_method='pgd', eps=eps)

        # 3) 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_fn,
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

