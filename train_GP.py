import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date, datetime
from utils.data_processing import normalize_and_save, normalize_std_scaler, load_and_normalize
from utils.models import FFNN_model
from utils.utils import load_json, save_acc_plot, save_loss_plot, name_date, name_time, name_to_dir

p = load_json('./params.json')
df = pd.read_csv(p.train_data_dir)

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

X  = df[p.feature_list].values.astype(np.float32)   # <-- float32로 강제 변환
y  = df[p.output_list].values.astype(np.float32)    # <-- float32로 강제 변환

print(X)

if SAVE_NORMALIZATION_FILE:
    scaler = normalize_and_save(X,time_flag=True)
    X = normalize_std_scaler(X, scaler)
else:
    X = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')

# print(X)
# print(y)

# 모델 정의 (예: 회로 손실 예측용 MLP)
inputs = tf.keras.Input(shape=(len(p.feature_list),))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(p.output_list))(x)
model = tf.keras.Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam()
lipschitz_lambda = 0.00025

@tf.function
def train_step(x, y):
    # 1) tape를 두 개 켜서, 파라미터와 입력에 대한 gradient를 모두 추적
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)   # 입력 x에 대한 gradient 추적
        y_pred = model(x, training=True)
        loss_mse = tf.reduce_mean((y_pred - y)**2)

        # 2) 입력 x에 대한 출력 gradient 계산
        grads_wrt_x = tape.gradient(y_pred, x)           # shape = (batch_size, n_features)
        grad_norm_sq = tf.reduce_mean(tf.reduce_sum(grads_wrt_x**2, axis=1))
        
        # 3) 최종 손실 = MSE + λ * Gradient Penalty
        loss = loss_mse + lipschitz_lambda * grad_norm_sq

    # 4) 모델 파라미터 업데이트
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 훈련 루프
for epoch in range(p.epochs):
    loss_value = train_step(X, y)
    if epoch % 1000 == 0:
        print('epoch:',epoch, ' ,loss:',loss_value)

model_name = name_time('DNN_DAB_est.h5')
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)
print(f"Saved model to {model_path}")