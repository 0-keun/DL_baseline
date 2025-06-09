import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from utils.data_processing import load_serial_data_from_csv
from utils.utils import save_loss_plot, save_acc_plot, name_date, name_time
from utils.models import LSTM_model

# 멀티-GPU 전략
strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("float32")


############################
##       get params       ##
############################

import json

with open('./params.json', 'r') as f:
    params = json.load(f)

TIME_STEPS = params["time_steps"]
BULK_SIZE = params["bulk_size"]
TRAIN_DATA_DIR = params["train_data_dir"]
EPOCHS        = params["epochs"]
BATCH_SIZE    = params["batch_size"]
FEATURE_LIST = params["feature_list"]
CLASSES_LIST = params["classes_list"]
CLASS_NUM = len(CLASSES_LIST)+1
FEATURE_NUM = len(FEATURE_LIST)

MODEL_DIR = name_date(default_name='model')


############################
##       Model setup      ##
############################

class Train_Model:
    def __init__(self, hidden_state_num, layer_num):
        # 하이퍼파라미터
        self.hidden_state_num = hidden_state_num
        self.layer_num = layer_num
        model_name = name_time(default_name=f'LSTM_h{hidden_state_num}_layer{layer_num}_class{CLASS_NUM}', ext='.h5')
        self.model_name = (f'./model/{MODEL_DIR}/{model_name}')

        # 데이터 로드
        self.X_input, self.y_output = load_serial_data_from_csv(TRAIN_DATA_DIR,FEATURE_LIST,CLASSES_LIST,BULK_SIZE,TIME_STEPS)

    def train_model(self, model):
        start_time = time.time()
        history = model.fit(
            self.X_input, self.y_output,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        training_loss_file_name = name_time(default_name='training_loss', ext='.png')
        training_accuracy_file_name = name_time(default_name='training_accuracy', ext='.png')
        save_loss_plot(history,loss_filepath=training_loss_file_name)
        save_acc_plot(history,acc_filepath=training_accuracy_file_name)
        
        model.save(self.model_name)

    def main(self):
        model = LSTM_model(self.hidden_state_num, CLASS_NUM, TIME_STEPS, FEATURE_NUM, self.layer_num)
        self.train_model(model)

if __name__ == "__main__":
    tm_500_2 = Train_Model(hidden_state_num=500, layer_num=2)
    tm_500_2.main()