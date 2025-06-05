import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from utils.data_processing import load_serial_data_from_csv
from utils.utils import save_loss_plot, save_acc_plot
from models.models import LSTM_model

# 멀티-GPU 전략
strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("float32")

import matplotlib.pyplot as plt

class_num = 9
time_steps = 5
feature_list = ['Vo:Measured voltage','IL:Measured current','Vin:Measured voltage']
classes_list = ['Open_1','Short_1','Short_2','Short_3','Short_4','Open_2','Open_3','Open_4']
bulk_size = 10
feature_num = len(feature_list)
train_data_dir = './dataset/train_data.csv'
epochs        = 100
batch_size    = 64


class Train_Model:
    def __init__(self, hidden_state_num, layer_num):
        # 하이퍼파라미터
        self.hidden_state_num = hidden_state_num
        self.layer_num = layer_num
        self.model_name    = (
            f'./model/LSTM_h{hidden_state_num}_layer{layer_num}_9class.h5'
        )

        # 데이터 로드
        data = load_serial_data_from_csv(train_data_dir,feature_list,classes_list,bulk_size,time_steps)

    def train_model(self, model):
        start_time = time.time()
        history = model.fit(
            self.X_input, self.y_output,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        save_loss_plot(history,loss_filepath='training_loss.png')
        save_acc_plot(history,acc_filepath='training_accuracy.png')
        print(f"Training of {self.model_name} took {time.time() - start_time:.1f} seconds")
        
        model.save(self.model_name)

    def main(self):
        model = LSTM_model(self.hidden_state_num, class_num, time_steps, feature_num, self.layer_num)
        self.train_model(model)

if __name__ == "__main__":
    tm_500_2 = Train_Model(hidden_state=500, num_layers=2)
    tm_500_2.main()