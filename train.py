import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date, datetime
from utils.data_processing import normalizae_and_save
from utils.models import FFNN_model
from utils.utils import load_json, save_acc_plot, save_loss_plot, name_date, name_time, name_to_dir

p = load_json('./params.json')
df = pd.read_csv(p.train_data_dir)

MODEL_DIR = name_to_dir(name='model',time_flag=True)

X = df[p.feature_list].values 
y = df[p.output_list].values  

scaler = normalizae_and_save(X)
X = scaler.transform(X)

model = FFNN_model(feature_num=len(p.feature_list), output_num=len(p.output_list))

history = model.fit(
    X, y,
    epochs=p.epochs,
    batch_size=p.batch_size,
    validation_split=0.2
)

save_loss_plot(history=history,loss_filename='loss.png',time_flag=True)

model_name = name_time('DNN_DAB_est.h5')
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)
print(f"Saved model to {model_path}")