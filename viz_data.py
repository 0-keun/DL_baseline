import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf
from utils.data_processing import load_serial_data_from_csv, normalize_and_save, add_normal_class, read_all_csv_to_np_list, make_sequence_dataset_specific_section, load_and_normalize, normalize_std_scaler
from utils.utils import save_loss_plot, save_acc_plot, name_to_dir, name_time, load_json
from utils.models import LSTM_model, transformer_model, SaveEveryNEpoch

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
        self.X_input, self.y_output = make_sequence_dataset_specific_section(p.train_data_dir,p.time_steps,p.feature_list,p.classes_list)
        self.X_input_noise, self.y_output_noise = make_sequence_dataset_specific_section(p.noise_data_dir,p.time_steps,p.feature_list,p.viz_classes_list)
        if SAVE_NORMALIZATION_FILE:
            features_data, _ = read_all_csv_to_np_list('./dataset/dataset_normal_250610',p.feature_list,p.classes_list,dim_reduction=True)
            scaler = normalize_and_save(np.squeeze(features_data),time_flag=True)
            self.X_input = normalize_std_scaler(self.X_input, scaler)
        else:
            self.X_input = load_and_normalize(self.X_input,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')
            self.X_input_noise = load_and_normalize(self.X_input_noise,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')

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
        model = LSTM_model(self.hidden_state_num, len(p.classes_list)+1, p.time_steps, len(p.feature_list), self.layer_num)
        self.train_model(model)

    def main_more(self):
        # ./model/model_250624/LSTM_h256_layer4_class3_104635 is the vanilla model
        model = model = tf.keras.models.load_model('./model/model_250624/LSTM_Vanilla_normal_104635.h5')
        self.train_model(model)

def process_3d_to_2d(X):
    """
    (n_samples, n_timesteps, n_features) 형태의 데이터를 받아 각 샘플의 타임스텝을 펼쳐서
    (n_samples, n_timesteps * n_features) 형태로 변환합니다.
    
    Parameters:
    X (numpy.ndarray): shape가 (n_samples, n_timesteps, n_features)인 3D 배열
    
    Returns:
    numpy.ndarray: shape가 (n_samples, n_timesteps * n_features)인 2D 배열
    """
    # 모든 타임스텝을 하나의 벡터로 결합
    X_2d = X.reshape(X.shape[0], -1)  # (n_samples, n_timesteps * n_features)
    
    return X_2d

def delete_data(arr, idx):
    # 인덱스 2(세 번째 값) 제거
    return np.delete(arr, idx, axis=2)


def viz_UMAP(predictions,predicted_classes):
    import matplotlib.pyplot as plt
    import umap
    print(len(predictions[0]))
    # UMAP을 사용하여 차원 축소
    umap_model = umap.UMAP(n_components=2)
    X_umap = umap_model.fit_transform(predictions)  # 예측 확률을 2D로 축소

    # 시각화
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=predicted_classes, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('UMAP on Predicted Probabilities')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

def viz_UMAP2D_6classes(predictions, predicted_classes, class_names=None, random_state=42, s=18, alpha=0.9):
    """
    predictions: (N, D) ndarray
    predicted_classes: (N,) 정수 배열/리스트, 값은 0~5
    class_names: 길이 6 리스트(옵션). 예: ['C0','C1','C2','C3','C4','C5']
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import umap

    X = np.asarray(predictions)
    y = np.asarray(predicted_classes).astype(int)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"샘플 수 불일치: X={X.shape[0]}, y={y.shape[0]}")
    uniq = np.unique(y)
    if np.any((uniq < 0) | (uniq > 5)):
        raise ValueError(f"predicted_classes는 0~5 범위여야 합니다. 현재 고유값: {uniq}")

    # UMAP 2D
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    X_umap = reducer.fit_transform(X)  # (N, 2)

    # 0/3 파랑, 1/4 주황, 2/5 초록 (3,4,5는 같은 계열의 밝은 톤)
    colors6 = [
        '#1f77b4', '#ff7f0e', '#2ca02c',  # 0,1,2 (진한 톤)
        '#6baed6', '#ffbb78', '#98df8a'   # 3,4,5 (밝은 톤)
    ]
    cmap = ListedColormap(colors6)
    boundaries = np.arange(-0.5, 6.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    # 그림
    plt.figure(figsize=(12, 9))
    sc = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=cmap, norm=norm, s=s, alpha=alpha)

    plt.xlabel('UMAP Component 1', fontsize=14)
    plt.ylabel('UMAP Component 2', fontsize=14)
    plt.title('UMAP 2D (6 classes, grouped colors)', fontsize=16)
    plt.tick_params(axis='both', labelsize=12)

    # 컬러바: 0~5 눈금
    cbar = plt.colorbar(sc, ticks=np.arange(0, 6))
    if class_names is not None:
        if len(class_names) != 6:
            raise ValueError("class_names 길이는 6이어야 합니다.")
        cbar.ax.set_yticklabels(class_names, fontsize=12)
    else:
        cbar.ax.set_yticklabels([str(i) for i in range(6)], fontsize=12)

    plt.tight_layout()
    plt.show()

def viz_UMAP_click(predictions, predicted_classes):
    import matplotlib.pyplot as plt
    import umap
    # UMAP을 사용하여 차원 축소 (2D)
    umap_model = umap.UMAP(n_components=2)  # n_components=2로 설정하여 2D로 축소
    X_umap = umap_model.fit_transform(predictions)  # 예측 확률을 2D로 축소
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=predicted_classes, cmap='viridis')

    # 색상 바 추가
    plt.colorbar(scatter)

    # 시각화 설정
    ax.set_title('UMAP on Predicted Probabilities')
    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')

    # 클릭 이벤트 처리 함수
    def onpick(event):
        # 클릭한 데이터 포인트의 인덱스
        ind = event.ind  
        index = ind[0]  # 첫 번째 클릭된 데이터 인덱스
        
        # 데이터 정보 출력
        print(f"Clicked point index: {index}")
        print(f"Coordinates of clicked point: {X_umap[index]}")
        print(f"Predicted class of clicked point: {predicted_classes[index]}")

        # 클릭된 위치에 텍스트로 몇 번째 데이터인지 표시
        ax.text(X_umap[index, 0], X_umap[index, 1], f'{index}', color='red', fontsize=12, ha='center')

        # 텍스트 업데이트 및 다시 그리기
        plt.draw()

    # 클릭 이벤트 연결
    fig.canvas.mpl_connect('pick_event', onpick)

    # 포인트에 대한 pickable 설정
    scatter.set_picker(True)

    # 시각화
    plt.show()

def viz_UMAP3D(predictions, predicted_classes):
    import matplotlib.pyplot as plt
    import umap

    # UMAP을 사용하여 차원 축소 (3D)
    umap_model = umap.UMAP(n_components=3)  # n_components=3으로 설정하여 3D로 축소
    X_umap = umap_model.fit_transform(predictions)  # 예측 확률을 3D로 축소

    # 3D 시각화
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')  # 3D 축 생성

    # scatter plot 생성
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=predicted_classes, cmap='viridis')

    # 시각화 설정
    ax.set_xlabel('UMAP Component 1', fontsize=16)  # 글씨 크기 조정
    ax.set_ylabel('UMAP Component 2', fontsize=16)  # 글씨 크기 조정
    ax.set_zlabel('UMAP Component 3', fontsize=16)  # 글씨 크기 조정
    ax.set_title('Distribution of data', fontsize=18)  # 제목 글씨 크기 조정
    # 숫자 레이블(눈금) 글씨 크기 조정
    ax.tick_params(axis='x', labelsize=16)  # x축 숫자 크기
    ax.tick_params(axis='y', labelsize=16)  # y축 숫자 크기
    ax.tick_params(axis='z', labelsize=16)  # z축 숫자 크기
    # 색상 바 추가
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=12)

    # 시각화
    plt.show()

def viz_UMAP3D_6classes(predictions, predicted_classes, class_names=None, random_state=42):
    """
    predictions: (N, D) ndarray
    predicted_classes: (N,) ndarray/list, 값은 0~5의 정수
    class_names: 길이 6의 리스트(옵션). 예: ['C0','C1','C2','C3','C4','C5']
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import umap

    predicted_classes = np.asarray(predicted_classes).astype(int)

    # 안전장치: 클래스 값이 0~5인지 확인
    uniq = np.unique(predicted_classes)
    if np.any((uniq < 0) | (uniq > 5)):
        raise ValueError(f"predicted_classes는 0~5 범위여야 합니다. 현재 고유값: {uniq}")

    # UMAP 3D 임베딩
    reducer = umap.UMAP(n_components=3) #, random_state=random_state)
    X_umap = reducer.fit_transform(predictions)  # (N, 3)

    # 6색 고정 팔레트 (구분 잘 되는 기본색)
    colors6 = [
    '#1f77b4', '#ff7f0e', '#2ca02c',  # 0,1,2 (진한 톤)
    '#6baed6', '#ffbb78', '#98df8a'   # 3,4,5 (같은 계열의 밝은 톤)
    ]
    cmap = ListedColormap(colors6)
    boundaries = np.arange(-0.5, 6.5, 1.0)  # [-0.5, 0.5, ..., 5.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    # 그림
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        X_umap[:, 0], X_umap[:, 1], X_umap[:, 2],
        c=predicted_classes, cmap=cmap, norm=norm, s=12, alpha=0.9
    )

    # 축/제목
    ax.set_xlabel('UMAP Component 1', fontsize=16)
    ax.set_ylabel('UMAP Component 2', fontsize=16)
    ax.set_zlabel('UMAP Component 3', fontsize=16)
    ax.set_title('Distribution of data (6 classes)', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)

    # 컬러바: 0~5 눈금 고정
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=12, ticks=np.arange(0, 6))
    if class_names is not None:
        if len(class_names) != 6:
            raise ValueError("class_names 길이는 6이어야 합니다.")
        cbar.ax.set_yticklabels(class_names, fontsize=14)
    else:
        cbar.ax.set_yticklabels([str(i) for i in range(6)], fontsize=14)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

def plot_overlaid_timeseries(X, y, alpha=0.08, linewidth=0.7, class_colors=None, show_means=False):
    """
    모든 시계열을 한 그래프에 겹쳐 그리고, 클래스별 색상을 구분해서 표시합니다.
    0-3, 1-4, 2-5 클래스끼리는 비슷한 색 계열 사용.
    """
    if X.ndim != 2:
        raise ValueError("X는 (n_series, T) 형태여야 합니다.")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y는 길이가 n_series인 1차원 배열이어야 합니다.")

    n_series, T = X.shape
    t = np.arange(T, dtype=float)

    # 기본 색상 (0/3 파랑, 1/4 주황, 2/5 초록)
    if class_colors is None:
        base_colors = {
            0: '#1f77b4',  # blue
            3: '#1f77b4',
            1: '#ff7f0e',  # orange
            4: '#ff7f0e',
            2: '#2ca02c',  # green
            5: '#2ca02c'
        }
        # 데이터에 존재하는 클래스만 매핑
        uniq = np.unique(y)
        class_colors = {c: base_colors.get(c, f'C{c}') for c in uniq}

    # NaN/Inf 처리
    if not np.isfinite(X).all():
        X = np.where(np.isfinite(X), X, np.nan)
        for i in range(n_series):
            xi = X[i]
            mask = np.isfinite(xi)
            if not mask.any():
                continue
            X[i, ~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), xi[mask])

    fig, ax = plt.subplots(figsize=(10, 6))

    for c, col in class_colors.items():
        mask = (y == c)
        if not np.any(mask):
            continue
        Xc = X[mask]
        k = Xc.shape[0]

        segs = np.empty((k, T, 2), dtype=float)
        segs[..., 0] = t
        segs[..., 1] = Xc

        lc = LineCollection(segs, colors=col, linewidths=linewidth, alpha=alpha)
        ax.add_collection(lc)

        if show_means:
            mean_curve = Xc.mean(axis=0)
            ax.plot(t, mean_curve, color=col, linewidth=2.0, alpha=0.9)

    ax.set_xlim(float(t.min()), float(t.max()))

    # 범례
    for c, col in class_colors.items():
        ax.plot([], [], color=col, label=f'class {c}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Overlaid Time Series by Class (Grouped Colors)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def trans_std(dataset):
    new_dataset = []
    for data in dataset:
        data_T = data.T
        std_0 = np.std(data_T[0])
        std_1 = np.std(data_T[1])
        new_dataset.append([std_0,std_1])

    return new_dataset

def mean_std(dataset):
    new_dataset = []
    for data in dataset:
        data_T = data.T
        mean_0 = np.mean(data_T[0])
        std_0 = np.std(data_T[0])
        mean_1 = np.mean(data_T[1])
        std_1 = np.std(data_T[1])
        new_dataset.append([mean_0,mean_1,std_0,std_1])

    return new_dataset

def edit_class(y_origin, y_noise):
    y_origin_zeros = np.hstack([y_origin, np.zeros((y_origin.shape[0], 3))])
    y_noise_zeros = np.hstack([np.zeros((y_noise.shape[0], 3)), y_noise])
    return  np.vstack([y_origin_zeros,y_noise_zeros])

if __name__ == "__main__":
    tm_256_4 = Train_Model(hidden_state_num=256, layer_num=4)
    X_input = delete_data(tm_256_4.X_input,2)
    X_input_noise = delete_data(tm_256_4.X_input_noise,2)
    # X_input = delete_data(X_input,0)
    # ms_X_input = mean_std(X_input)
    # ms_X_input_noise = mean_std(X_input_noise)

    ms_X_input = trans_std(X_input)
    ms_X_input_noise = trans_std(X_input_noise)
    total_data = ms_X_input + ms_X_input_noise
    total_label = edit_class(tm_256_4.y_output,tm_256_4.y_output_noise)
    # print(np.array(total_data))
    # print(total_label)
    # viz_UMAP(process_3d_to_2d(X_input),predicted_classes = np.argmax(tm_256_4.y_output, axis=1))
    viz_UMAP2D_6classes(total_data,predicted_classes = np.argmax(total_label, axis=1),class_names=['open','short','normal','open_with_noise','short_with_noise','normal_with_noise'])

    # plot_overlaid_timeseries(process_3d_to_2d(tm_256_4.X_input), np.argmax(tm_256_4.y_output, axis=1))