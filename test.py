import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import re
from utils.data_processing import load_serial_data_from_csv,make_sequence_dataset, add_normal_class, load_and_normalize
from utils.utils import load_json, get_confusion_mat
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import json

with open('./params.json', 'r') as f:
    params = json.load(f)

p = load_json()

def get_params(filename):
    # 1) basename만 뽑아내고 싶으면 pathlib 사용
    from pathlib import Path
    stem = Path(filename).stem      # → 'LSTM_h10_layer3'

    # 2) 정규표현식 패턴
    pattern = r'LSTM_h(\d+)_layer(\d+)\_class(\d+)_(\d+).h5$'

    m = re.search(pattern, filename)
    if m:
        num1 = int(m.group(1))   # h 뒤 숫자
        num2 = int(m.group(2))   # layer 뒤 숫자
        num3 = int(m.group(3))
        return num1, num2, num3

class Tester():
    def __init__(self, model_name):
        self.model = load_model(model_name)
        self.hidden_state, self.num_layer, _ = get_params(model_name)
        self.X_input, self.y_output = make_sequence_dataset(p.test_data_dir,p.time_steps,p.feature_list,p.classes_list)
        self.X_input = load_and_normalize(self.X_input,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')
        self.y_output = add_normal_class(self.y_output)

        list = [0]*(len(p.classes_list)+1)
        for output in self.y_output:
            list += output
        print("sample distribution by class:", list)  

    def main(self):
        # 예측 수행
        y_pred = self.model.predict(self.X_input)

        # 1차원 레이블로 변환
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_output, axis=1)

        # 정확도(Accuracy) 계산
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # # 분류 보고서
        # print("\nClassification Report:")
        # print(classification_report(y_true_classes, y_pred_classes))

        # 혼동 행렬
        get_confusion_mat(y_true_classes, y_pred_classes,time_flag=True)

        # # 실행 시간 출력
        # execution_time = end_time - start_time
        # print("The prediction took", execution_time, "seconds to complete")

test_500_3 = Tester('./model/model_250610/LSTM_h500_layer3_class9_230620'+'.h5')

test_500_3.main()
