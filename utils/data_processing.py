from normalization import instance_normalize
from pandas import read_csv
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_serial_data_from_csv(input_file,feature_list,classes_list,bulk_size,time_steps):
    '''
    Load sequential data from an input file that contains multiple stacked episodes
    '''
    df = read_csv(input_file, index_col=None)
    data = df[feature_list].values
    cls = np.full(len(df), 0, dtype=int)  # 0=정상으로 초기화

    for i in range(len(classes_list)):
        cls[df[classes_list[i]] > 0] = i+1

    labels = to_categorical(cls, num_classes=len(classes_list))  # shape: (N, 9)

    # 나머지 시계열 준비 및 정규화 등은 동일하게 유지
    Xs, Ys = [], []
    ep_num = len(data) // bulk_size
    for ep in range(ep_num):
        start = ep * bulk_size
        for i in range(bulk_size - time_steps):
            instance_start = start + i
            instance_end = instance_start + time_steps
            Xs.append(data[instance_start:instance_end])
            Ys.append(labels[instance_end])

    X_input = np.array(Xs)
    y_output = np.array(Ys)

    # 각 시퀀스별 정규화
    norm_data = [instance_normalize(instance, method="minmax") for instance in X_input]
    X_input = np.array(norm_data)

    list = [0]*len(classes_list)
    for output in y_output:
        list += output
    print("sample distribution by class:", list)

    return X_input, y_output