import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import re
import os
from utils.data_processing import load_serial_data_from_csv,make_sequence_data, add_normal_class, load_and_normalize
from utils.utils import load_json, get_confusion_mat
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import json

p = load_json('./params_3F.json')
fname = '/data_f_6_t_1.165007.csv'

def get_params(filename, pattern = r'LSTM_h(\d+)_layer(\d+)\_class(\d+)_(\d+).h5$'):
    # 1) basename만 뽑아내고 싶으면 pathlib 사용
    from pathlib import Path
    stem = Path(filename).stem      # → 'LSTM_h10_layer3'

    m = re.search(pattern, filename)
    if m:
        num1 = int(m.group(1))   # h 뒤 숫자
        num2 = (m.group(2))   # layer 뒤 숫자
        return num1, num2
    

class Tester():
    def __init__(self, model_name):
        self.model = load_model(model_name)
        # self.hidden_state, self.num_layer, _ = get_params(model_name)
        # self.X_input, self.y_output = make_sequence_data(file_path,p.time_steps,p.feature_list,p.classes_list)
        # self.X_input = load_and_normalize(self.X_input,'./scaler/mean_135845.npy','./scaler/scale_135845.npy')
        # self.y_output = add_normal_class(self.y_output)

        # list = [0]*(len(p.classes_list)+1)
        # for output in self.y_output:
        #     list += output
        # print("sample distribution by class:", list)  

    def reset(self, file_path):
        self.X_input, self.y_output = make_sequence_data(file_path,p.time_steps,p.feature_list,p.classes_list)
        self.filename = os.path.basename(file_path)
        self.class_9F, _ = get_params(self.filename,r'data_f_(\d+)_t_(\d+(?:\.\d+)?).csv$')
        self.X_input = load_and_normalize(self.X_input,'./scaler/mean_135845.npy','./scaler/scale_135845.npy')
        self.y_output = add_normal_class(self.y_output)

    def main(self,wrong_list,class_list, list):
        # 예측 수행
        y_pred = self.model.predict(self.X_input, verbose=0)

        # 1차원 레이블로 변환
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_output, axis=1)

        # 정확도(Accuracy) 계산
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        # print(f"Accuracy: {accuracy * 100:.2f}%")
        cnt = 0
        for i in range(len(y_true_classes)):
            if not y_pred_classes[i] == y_true_classes[i]:
                wrong_list[i] += 1
                cnt += 1
        if cnt >= 10:
            class_list[self.class_9F] += 1
            list.append(self.filename)
        # print(wrong_list)
        return wrong_list, class_list, list

test_500_3 = Tester('./model/model_250617/LSTM_h256_layer4_class3_2041121000.h5')

w_list = np.zeros(160)
c_list = np.zeros(9)
list = []
for fname in os.listdir(p.test_data_dir):
    test_500_3.reset(p.test_data_dir+'/'+fname)
    w_list, c_list, list = test_500_3.main(w_list,c_list, list)
print(f"w_list = {w_list}")
print(f"c_list = {c_list}")
print(list)
