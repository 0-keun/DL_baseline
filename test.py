import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import re
from utils.data_processing import load_serial_data_from_csv, name_to_dir, load_and_normalize
from utils.models import FFNN_model
from utils.utils import load_json, save_loss_plot, name_time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

import json


# test_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# def get_params(filename):
#     # 1) basename만 뽑아내고 싶으면 pathlib 사용
#     from pathlib import Path
#     stem = Path(filename).stem      # → 'LSTM_h10_layer3'

#     # 2) 정규표현식 패턴
#     pattern = r'LSTM_h(\d+)_layer(\d+)\_class(\d+).h5$'

#     m = re.search(pattern, filename)
#     if m:
#         num1 = int(m.group(1))   # h 뒤 숫자
#         num2 = int(m.group(2))   # layer 뒤 숫자
#         num3 = int(m.group(3))
#         return num1, num2, num3

p = load_json('./params.json')
df = pd.read_csv(p.test_data_dir)

class Tester():
    def __init__(self, model_name):
        # self.hidden_state, self.num_layer, _ = get_params(model_name)
        X = df[p.feature_list].values 
        self.X_input = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')
        self.y_output = df[p.output_list].values 

        self.model = load_model(model_name)

    # def get_confusion_mat(self, y_true, y_pred):
    #     # 혼동 행렬
    #     cm=confusion_matrix(y_true, y_pred)
    #     # print("\nConfusion Matrix:")
    #     # print(cm)
    #     txt_name = "./confusion_matrix/confusion_matrix"+str(self.hidden_state)+"_"+str(self.num_layer)+".csv"
    #     np.savetxt(txt_name, cm, fmt='%d', delimiter=',')

    def main(self):
        # 예측 수행
        y_pred = self.model.predict(self.X_input)
        np.set_printoptions(suppress=True, precision=6)
        # print(f"Actual: {self.y_output}")
        # print(f"Pred: {y_pred}")
        for i in range(min(5, len(y_pred))):
            print(f"[{i}] Actual: {self.y_output[i]}")
            print(f"[{i}] Pred: {y_pred[i]}")
            for j in range(len(self.y_output[i])):
                print((y_pred[i][j] - self.y_output[i][j])*100/self.y_output[i][j])


            percent_error = np.abs(y_pred[i] - self.y_output[i]) * 100 / np.abs(self.y_output[i])
            mean_percent_error = np.mean(percent_error)
            print(f"[{i}] Percent Error: {mean_percent_error:.2f}%")

        percent_error = np.abs(y_pred - self.y_output) * 100 / np.abs(self.y_output)
        mean_percent_error = np.mean(percent_error)
        print(f"평균 Percent Error: {mean_percent_error:.2f}%")

test = Tester('./model/model_250613/DNN_DAB_est_162956.h5')

test.main()