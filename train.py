import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from utils.data_processing import load_serial_data_from_csv, normalize_and_save, add_normal_class, read_all_csv_to_np_list, make_sequence_dataset, load_and_normalize, normalize_std_scaler
from utils.utils import save_loss_plot, save_acc_plot, name_to_dir, name_time, load_json
from utils.models import LSTM_model, transformer_model

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("float32")


############################
##       get params       ##
############################

import json

with open('./params.json', 'r') as f:
    params = json.load(f)

p = load_json()

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = True


############################
##       Model setup      ##
############################

class Train_Model:
    def __init__(self, hidden_state_num, layer_num):
        # 하이퍼파라미터
        self.hidden_state_num = hidden_state_num
        self.layer_num = layer_num
        model_name = name_time(default_name=f'LSTM_h{hidden_state_num}_layer{layer_num}_class{len(p.classes_list)+1}.h5')
        self.model_filepath = MODEL_DIR+model_name

        # 데이터 로드
        self.X_input, self.y_output = make_sequence_dataset(p.train_data_dir,p.time_steps,p.feature_list,p.classes_list)
        if not SAVE_NORMALIZATION_FILE:
            features_data, _ = read_all_csv_to_np_list('./dataset/dataset_normal_250610',p.feature_list,p.classes_list,dim_reduction=True)
            scaler = normalize_and_save(np.squeeze(features_data),time_flag=True)
            self.X_input = normalize_std_scaler(self.X_input, scaler)
        else:
            self.X_input = load_and_normalize(self.X_input,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')
        self.y_output = add_normal_class(self.y_output)

        list = [0]*(len(p.classes_list)+1)
        for output in self.y_output:
            list += output
        print("sample distribution by class:", list)  

    def train_model(self, model):
        history = model.fit(
            self.X_input, self.y_output,
            epochs=p.epochs,
            batch_size=p.batch_size,
            verbose=1
        )

        save_loss_plot(history,loss_filename='training_loss.png',time_flag=True)
        save_acc_plot(history,acc_filename='training_accuracy.png',time_flag=True)
        
        model.save(self.model_filepath)

    def train_model_transformer(self, model):
        history = model.fit(
            self.X_input, self.y_output,
            epochs=p.epochs,
            batch_size=p.batch_size,
            verbose=1
        )
        save_loss_plot(history,loss_filepath='training_loss.png',time_flag=True)
        save_acc_plot(history,acc_filepath='training_accuracy.png',time_flag=True)
        
        model.save(self.model_filepath)

    def main(self):
        model = LSTM_model(self.hidden_state_num, len(p.classes_list)+1, p.time_steps, len(p.feature_list), self.layer_num)
        self.train_model(model)

if __name__ == "__main__":
    tm_500_3 = Train_Model(hidden_state_num=500, layer_num=3)
    tm_500_3.main()