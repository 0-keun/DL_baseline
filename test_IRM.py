#!/usr/bin/env python3
"""
Test script for IRM-based regression model.

Usage:
    python test_IRM.py \
        --model-path model.pth \
        --test-x test_X.npy \
        --test-y test_y.npy \
        --input-dim 100 \
        --hidden-dim 256 \
        --feature-dim 64 \
        [--output-dim 1] [--no-cuda]
"""
import argparse
import torch
import numpy as np
from train_IRM import IRMRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.data_processing import load_serial_data_from_csv, name_to_dir, load_and_normalize
from utils.models import FFNN_model
from utils.utils import load_json, save_loss_plot, name_time

p = load_json('./params.json')
df = pd.read_csv(p.test_data_dir)

class Tester:
    def __init__(self, model_path, input_dim, hidden_dim, feature_dim, output_dim, device):

        X = df[p.feature_list].values 
        self.X_input = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')
        self.y_output = df[p.output_list].values

        # 디바이스 설정
        self.device = device
        # 모델 구조 초기화 및 state_dict 불러오기
        self.model = IRMRegressor(input_dim, hidden_dim, feature_dim, output_dim).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        # self.model = state.to(self.device)
        # state_dict 방식일 경우 바로 load
        if isinstance(state, dict):
            self.model.load_state_dict(state)
        else:
            # 전체 모델 저장 방식일 경우
            self.model = state.to(self.device)
        
        self.model.eval()

    def evaluate(self, X_test, y_test, print_n=None):
        # NumPy 배열을 Tensor로 변환
        X = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # 예측 및 MSE 계산
        with torch.no_grad():
            preds, _ = self.model(X)

        print(preds)
        mse = torch.nn.functional.mse_loss(preds, y, reduction='mean').item()
        print(f"Test MSE: {mse:.6f}")

        # Tensor → NumPy 변환
        preds_np = preds.cpu().numpy()
        y_np     = y.cpu().numpy()

        # 원하면 앞 print_n개만 출력
        if print_n is not None:
            print(f"\n첫 {print_n}개 예측값 (y_pred) vs 실제값 (y_true):")
            for i in range(min(print_n, len(preds_np))):
                print(f"idx {i:03d}:") 
                print(f"pred = {preds_np[i]}")
                print(f"true = {y_np[i]}")
        percent_error = np.abs(preds_np - y_np) * 100 / np.abs(y_np)
        mean_percent_error = np.mean(percent_error)
        print(f"평균 Percent Error: {mean_percent_error:.2f}%")
        # 전체를 보고 싶으면 DataFrame으로 정리해서 반환·저장 가능
        # df_res = pd.DataFrame({
        #     'y_true': y_np,
        #     'y_pred': preds_np,
        # })
        return mse#, df_res


def load_numpy_data(path_x, path_y):
    X = np.load(path_x)
    y = np.load(path_y)
    return X, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 테스트
    tester = Tester(
        model_path='./model/model_250613/DNN_DAB_est_e100000.pth',
        input_dim=len(p.feature_list),
        hidden_dim=256,
        feature_dim=256,
        output_dim=len(p.output_list),
        device=device
    )
    mse = tester.evaluate(tester.X_input, tester.y_output, print_n=10)


if __name__ == "__main__":
    main()
