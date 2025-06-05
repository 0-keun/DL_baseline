import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import re
from utils.data_processing import load_serial_data_from_csv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

import json

with open('./params/params.json', 'r') as f:
    params = json.load(f)

TIME_STEPS = params["time_steps"]
BULK_SIZE = params["bulk_size"]
TEST_DATA_DIR = params["test_data_dir"]
EPOCHS        = params["epochs"]
BATCH_SIZE    = params["batch_size"]
FEATURE_LIST = params["feature_list"]
CLASSES_LIST = params["classes_list"]
CLASS_NUM = len(CLASSES_LIST)+1
FEATURE_NUM = len(FEATURE_LIST)

def get_params(filename):
    # 1) basename만 뽑아내고 싶으면 pathlib 사용
    from pathlib import Path
    stem = Path(filename).stem      # → 'LSTM_h10_layer3'

    # 2) 정규표현식 패턴
    pattern = r'LSTM_h(\d+)_layer(\d+)\_9class.h5$'

    m = re.search(pattern, filename)
    if m:
        num1 = int(m.group(1))   # h 뒤 숫자
        num2 = int(m.group(2))   # layer 뒤 숫자
        return num1, num2

class Tester():
    def __init__(self, model_name):
        self.hidden_state, self.num_layer = get_params(model_name)

        self.X_input, self.y_output = load_serial_data_from_csv(TEST_DATA_DIR,FEATURE_LIST,CLASSES_LIST,BULK_SIZE,TIME_STEPS)
        self.model = load_model(model_name)
    
    def get_confusion_mat(self, y_true, y_pred):
        # 혼동 행렬
        cm=confusion_matrix(y_true, y_pred)
        # print("\nConfusion Matrix:")
        # print(cm)
        txt_name = "./confusion_matrix/confusion_matrix"+str(self.hidden_state)+"_"+str(self.num_layer)+".csv"
        np.savetxt(txt_name, cm, fmt='%d', delimiter=',')

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
        self.get_confusion_mat(y_true_classes, y_pred_classes)

        # # 실행 시간 출력
        # execution_time = end_time - start_time
        # print("The prediction took", execution_time, "seconds to complete")




test_500_2 = Tester('./model/LSTM_h500_layer2_9class'+'.h5')

test_500_2.main()
