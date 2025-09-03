import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from utils.utils import load_json
from torch.nn.utils import spectral_norm
from educlidean_distance import comp_X_Z_plot

# ======================
# 설정
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 원래 코드의 FeatureExtractor 그대로 사용
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=255, output_dim=32, SN_flag = False):
        super().__init__()
        if not SN_flag:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.net = nn.Sequential(
                spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.ReLU(),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                spectral_norm(nn.Linear(hidden_dim, output_dim))
            )

    def forward(self, x):
        return self.net(x)

def main():
    # ----------------------
    # 1) 설정/데이터 로딩
    # ----------------------
    p = load_json('./params.json')
    df_test = pd.read_csv(p.test_data_dir)
    X_new = df_test[p.feature_list].values  # (N, D_in)
    SN_FLAG = True
    if SN_FLAG:
        MODEL_DIR = "./model/dkl_model_noSVD"
    else:
        MODEL_DIR = "./model/dkl_model_noSN"

    # 스케일러 로딩 & 변환
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_scaled = scaler.transform(X_new)

    # 텐서로 변환
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # 몇 개의 feature_extractor가 저장되어 있는지 확인 (feature_0.pth, feature_1.pth, ...)
    feature_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("feature_") and f.endswith(".pth")])
    if not feature_files:
        raise FileNotFoundError(f"No feature_*.pth files found in {MODEL_DIR}")

    print(f"[INFO] Found {len(feature_files)} feature extractors: {feature_files}")

    # 출력 폴더
    os.makedirs("features_only", exist_ok=True)

    # ----------------------
    # 2) i별 FeatureExtractor 로드 → 임베딩 계산
    # ----------------------
    all_Z = []
    with torch.no_grad():
        for fpath in feature_files:
            # i 인덱스 추출
            stem = os.path.splitext(fpath)[0]           # "feature_0"
            idx_str = stem.split("_")[-1]               # "0"
            i = int(idx_str)

            # FeatureExtractor 초기화 및 가중치 로드
            feat = FeatureExtractor(input_dim=X_tensor.shape[1], SN_flag=SN_FLAG).to(device)
            state = torch.load(os.path.join(MODEL_DIR, fpath), map_location=device)
            feat.load_state_dict(state)
            feat.eval()

            # 임베딩 계산 (N, 32)
            Z = feat(X_tensor)                 # torch.Tensor
            Z = Z.detach().cpu().numpy()       # (N, 32)

            # NaN 방지 (혹시 모를 수 있음)
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

            # # 저장 (CSV)
            # out_csv = os.path.join("features_only", f"features_{i}.csv")
            # pd.DataFrame(Z, columns=[f"z{i}_{k}" for k in range(Z.shape[1])]).to_csv(out_csv, index=False)
            # print(f"[OK] Saved features for i={i} → {out_csv} | shape: {Z.shape}")

            all_Z.append(Z)

    # 선택: 모든 i의 임베딩을 축으로 쌓아 3차원 배열로 저장 (num_i, N, 32)
    if all_Z:
        print(f"X_scaled: {len(X_scaled[0])}")
        # print(f"all_Z: {all_Z}")
        print(f"all_Z_len: {len(all_Z)}")
        comp_X_Z_plot(X_scaled, all_Z, feature_num=X_tensor.shape[1])
        # Z_stack = np.stack(all_Z, axis=0)
        # np.save(os.path.join("features_only", "features_stack.npy"), Z_stack)
        # print(f"[OK] Saved stacked features → features_only/features_stack.npy | shape: {Z_stack.shape}")

if __name__ == "__main__":
    main()







# import numpy as np

# data_before, data_after : shape (3, 250)
# 예시로 data라는 변수 안에 적용 전/후 결과를 따로 넣었다고 가정

# 유클리드 거리 계산 함수
# def euclidean_distance(a, b):
#     return np.linalg.norm(a - b)

# # 각 샘플별 거리 계산
# distances = []
# for i in range(data_before.shape[0]):
#     d = euclidean_distance(data_before[i], data_after[i])
#     distances.append(d)

# print("각 샘플별 Euclidean distance:", distances)
# print("평균 distance:", np.mean(distances))
