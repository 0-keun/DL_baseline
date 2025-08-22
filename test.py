import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import re
from utils.data_processing import load_serial_data_from_csv,make_sequence_dataset, add_normal_class, load_and_normalize
from utils.utils import load_json, get_confusion_mat_size
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import json

p = load_json('./params_3F.json')

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
        self.model_name = model_name
        # self.hidden_state, self.num_layer, _ = get_params(model_name)
        self.X_input, self.y_output = make_sequence_dataset(p.test_data_dir,p.time_steps,p.feature_list,p.classes_list)
        self.X_input = load_and_normalize(self.X_input,'./scaler/mean_135845.npy','./scaler/scale_135845.npy')
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

        # 혼동 행렬
        trained_data = p.train_data_dir.split('_')[-1]
        tested_data = p.test_data_dir.split('_')[-1]
        model_fname = self.model_name.split('/')[-1].split('_')[-1].split('.')[0]
        model_fdir = self.model_name.split('/')[-2].split('-')[-1]
        get_confusion_mat_size(y_true_classes, y_pred_classes, tested_model=model_fdir+model_fname, trained_data=trained_data, tested_data=tested_data, time_flag=True)

def viz_data(predictions,predicted_classes):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # t-SNE를 사용하여 차원 축소
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(predictions)  # 예측 확률을 2D로 축소

    # 시각화
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=predicted_classes, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE on Predicted Probabilities')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def viz_UMAP(predictions,predicted_classes):
    import matplotlib.pyplot as plt
    import umap
    # UMAP을 사용하여 차원 축소
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(predictions)  # 예측 확률을 2D로 축소

    # 시각화
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=predicted_classes, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('UMAP on Predicted Probabilities')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

test_Vanilla_normal = Tester('./model/model_250624/LSTM_Vanilla_normal_104635.h5')
test_Vanilla_noise  = Tester('./model/model_250716/LSTM_Vanilla_noise_102601.h5')
test_ADV_normal     = Tester('./model/model_250715/LSTM_ADV_normal_105257.h5')
test_ADV_noise      = Tester('./model/model_250714/LSTM_ADV_noise_143023.h5')

# latent_output = test_Vanilla_normal.model.predict(test_Vanilla_normal.X_input)
# print(latent_output)
# predicted_classes = np.argmax(test_Vanilla_normal.y_output, axis=1)
# print(predicted_classes)
# viz_UMAP(latent_output,predicted_classes)
test_Vanilla_normal.main()
test_Vanilla_noise.main()
test_ADV_normal.main()    
test_ADV_noise.main()