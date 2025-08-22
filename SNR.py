import pandas as pd
import numpy as np
import os
from utils.data_processing import load_features_data_from_csv
from utils.utils import load_json, name_to_dir

# 설정값
snr_db      = 13.0      
snr_linear  = 10 ** (snr_db / 20)                       # 원하는 SNR 값 (dB 단위)
csv_dir     = './dataset/dataset_steady/'          # 원본 데이터 CSV 파일 경로
output_dir  = './dataset/dataset_steady_noise/'    # 노이즈 추가된 데이터 저장 경로
mean_path   = './scaler/scaler_250616/mean_142455.npy'  # StandardScaler.mean_ 을 저장한 .npy 파일
scale_path  = './scaler/scaler_250616/scale_142455.npy' # StandardScaler.scale_ 을 저장한 .npy 파일

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

mean  = np.load(mean_path).astype(np.float32)   # shape (n_features,)
scale = np.load(scale_path).astype(np.float32)  # shape (n_features,)

noise_std = scale / snr_linear

p = load_json(file_name='./params_3F.json')

# 1) 데이터 로드
for fname in os.listdir(csv_dir):
    if fname.endswith('.csv'):
        df = pd.read_csv(csv_dir+fname, index_col=None)

        mean  = np.load(mean_path).astype(np.float32)   # shape (n_features,)
        scale = np.load(scale_path).astype(np.float32)  # shape (n_features,)

        scale_series    = pd.Series(scale, index=p.feature_list)
        noise_std_series = scale_series / snr_linear
        for col in p.feature_list:
            sigma = noise_std_series[col]
            df[col] = df[col].astype(np.float32) + np.random.normal(0.0, sigma, size=len(df)).astype(np.float32)

        name, _ ,ext = fname.split('.')
        name = name+'.'+_
        ext = '.'+ext
        df.to_csv(output_dir+name+'_'+str(snr_db)+ext, index=False)

print(f"원본 데이터에 SNR={snr_db}dB에 맞춰 노이즈를 추가하고 '{output_dir+fname+'_'+str(snr_db)}'로 저장했습니다.")