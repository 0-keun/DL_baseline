import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date, datetime
from utils.data_processing import normalize_and_save, normalize_std_scaler, load_and_normalize
from utils.models import FFNN_model
from utils.utils import load_json, save_acc_plot, save_loss_plot, name_date, name_time, name_to_dir
from tensorflow.keras.callbacks import Callback

# n번에 한 번씩 출력하는 콜백 정의
class PrintEveryNEpoch(Callback):
    def __init__(self, n=10):
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:  # epoch은 0부터 시작하므로 +1
            logs = logs or {}
            print(
                f"Epoch {epoch+1}: "
                f"loss = {logs.get('loss'):.4f}, "
                f"val_loss = {logs.get('val_loss'):.4f}"
            )

p = load_json('./params.json')
df = pd.read_csv(p.train_data_dir)

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

X = df[p.feature_list].values 
y = df[p.output_list].values  

if SAVE_NORMALIZATION_FILE:
    scaler = normalize_and_save(X,time_flag=True)
    X = normalize_std_scaler(X, scaler)
else:
    X = load_and_normalize(X,'./scaler/scaler_250904/mean_152528.npy','./scaler/scaler_250904/scale_152528.npy')

# print(X)
# print(y)

model = FFNN_model(feature_num=len(p.feature_list), output_num=len(p.output_list))

# n 값 설정
N_PRINT = 500  # 5 epoch마다 출력

try:
    history = model.fit(
        X, y,
        epochs=p.epochs,
        batch_size=p.batch_size,
        validation_split=0.2,
        callbacks=[PrintEveryNEpoch(n=N_PRINT)],
        verbose=0
    )

    save_loss_plot(history=history,loss_filename='loss.png',time_flag=True)

    model_name = name_time('DNN_DAB_est.h5')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")

except KeyboardInterrupt:
    print("\n Training is terminated by user. The current model is saved...")
    model_name = name_time('DNN_DAB_est.h5')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")