import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from utils.data_processing import load_and_normalize
from GPA_train import GPAOnlyModel
from utils.utils import load_json

import numpy as np

# -------------------------
# 1. 파라미터 및 설정 로드
# -------------------------
p = load_json('./params.json')

# -------------------------
# 2. 테스트 데이터 로딩 및 정규화
# -------------------------
df = pd.read_csv(p.test_data_dir)
X_test = df[p.feature_list].values
y_test = df[p.output_list].values

y_0 = []
for y_instance in y_test:
    y_0.append(y_instance[0])

# 정규화 (train 시 저장해둔 scaler 사용)
X_test = load_and_normalize(
    X_test,
    './scaler/scaler_250612/mean_213605.npy',
    './scaler/scaler_250612/scale_213605.npy'
)

X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_0, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=1, shuffle=False)

# -------------------------
# 3. 모델 구성 및 불러오기
# -------------------------
model = GPAOnlyModel(
    input_dim=len(p.feature_list),
    rff_dim=256,
    output_dim=1 #len(p.output_list)
)

model_path = './model/model_250801/gpa_model.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

# -------------------------
# 4. 테스트 실행
# -------------------------
all_means = []
all_stds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        if len(x_batch) == 1:
            mean, std = model(x_batch)
            all_means.append(mean.numpy())
            all_stds.append(std.numpy())
            all_targets.append(y_batch.numpy().reshape(-1, y_batch.shape[-1]))  # 명시적으로 2D로 reshape
        
    # for x_batch, y_batch in test_loader:
    #     mean, std = model(x_batch)
    #     all_means.append(mean.numpy())
    #     all_stds.append(std.numpy())
    #     all_targets.append(y_batch.numpy())

# -------------------------
# 5. 결과 저장 및 출력
# -------------------------
mean_preds = np.vstack(all_means)
std_preds = np.vstack(all_stds)
targets = np.vstack(all_targets)

# 예시 출력
for i in range(min(5,len(targets))):
    print(f"Target: {targets[i]}, Predicted mean: {mean_preds[i]}, Std: {std_preds[i]}")

# print(mean_preds,len(mean_preds))
percent_error = np.abs(y_0 - mean_preds) * 100 / np.abs(mean_preds)
absolute_error = np.abs(y_0 - mean_preds)
mean_percent_error = np.mean(percent_error)
mean_absolute_error = np.mean(absolute_error)
print(f"평균 Percent Error: {mean_percent_error:.2f}%")
print(f"평균 Absolute Error: {mean_absolute_error:.2f}")

# print(y_0)

import matplotlib.pyplot as plt
import os
def plot_predictions(y_true, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label='Ground Truth')
    plt.plot(y_pred, label='Prediction')
    plt.title(f"Output: Prediction vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"output_prediction.png"))
    plt.close()

plot_predictions(np.array(y_0), mean_preds, output_dir='plots_GPA')

# # (선택) 전체 결과를 CSV로 저장
# result_df = pd.DataFrame({
#     **{f"y_true_{i}": targets[:len, i] for i in range(targets.shape[1])},
#     **{f"y_pred_{i}": mean_preds[:, i] for i in range(mean_preds.shape[1])},
#     **{f"y_std_{i}": std_preds[:, i] for i in range(std_preds.shape[1])}
# })
# result_df.to_csv("./results/test_predictions.csv", index=False)
