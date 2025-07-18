import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from utils.data_processing import load_serial_data_from_csv, normalize_and_save, add_normal_class, read_all_csv_to_np_list, make_sequence_dataset, load_and_normalize, normalize_std_scaler
from utils.utils import save_loss_plot, save_acc_plot, name_to_dir, name_time, load_json
from utils.models import LSTM_model_ADV, transformer_model, SaveEveryNEpoch

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("float32")


############################
##       get params       ##
############################

import json

p = load_json(file_name='./params_3F.json')

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False


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
        if SAVE_NORMALIZATION_FILE:
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
            verbose=1,
            callbacks=[SaveEveryNEpoch(save_path=self.model_filepath, interval=10)]
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
        model = LSTM_model_ADV(self.hidden_state_num, len(p.classes_list)+1, p.time_steps, len(p.feature_list), self.layer_num, eps=0.08)
        self.train_model(model)

    def main_more(self):
        model = tf.keras.models.load_model('./model/model_250716/LSTM_ADV2_normal_214418.h5')
        self.train_model(model)

if __name__ == "__main__":
    tm_256_4 = Train_Model(hidden_state_num=256, layer_num=4)
    tm_256_4.main_more()