import os
import glob
import pandas as pd
from utils.data_processing import load_classes_data_from_csv
from utils.utils import load_json

import json

with open('./params.json', 'r') as f:
    params = json.load(f)

p = load_json()

# 1. 폴더 경로 지정
folder_path = './dataset/dataset_added'  # 실제 경로로 변경하세요

# 2. 폴더 내 모든 CSV 파일 리스트 가져오기
open_cols = ['open1:Output','open2:Output','open3:Output','open4:Output']
short_cols = ['short1:Output','short2:Output','short3:Output','short4:Output']

def add_new_class(dir_name,cols,new_class):
    csv_files = glob.glob(os.path.join(dir_name, '*.csv'))
    for file in csv_files:
        df = pd.read_csv(file)
        df[new_class] = df.loc[:, cols].any(axis=1).astype(int)
        df.to_csv(file)

    print(f"New class {new_class} is successfully added")

add_new_class(folder_path, open_cols, 'OPEN')
add_new_class(folder_path, short_cols, 'SHORT')