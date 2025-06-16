import torch
import torch.nn as nn
import pandas as pd
import os
from utils.data_processing import name_time, name_to_dir, normalize_and_save, normalize_std_scaler, load_and_normalize
from utils.utils import load_json

p = load_json('./params.json')
df = pd.read_csv(p.train_data_dir)
MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

# 2) Vout 레벨 목록
vout_levels = [300.0, 350.0, 400.0]

# 3) environments 리스트 생성
environments = []
for v in vout_levels:
    sub = df[df['Vout'] == v]
    data = sub[p.feature_list].values
    if SAVE_NORMALIZATION_FILE:
        scaler = normalize_and_save(data,time_flag=True)
        X_e = normalize_std_scaler(data, scaler)
    else:
        X_e = load_and_normalize(data,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')
    X_e = torch.tensor(X_e, dtype=torch.float32)
    print(X_e)
    y_e = torch.tensor(sub[p.output_list].values,  dtype=torch.float32)
    environments.append((X_e, y_e))
    # print('env: ', environments)

# X = df[p.feature_list].values 
# y = df[p.output_list].values  

# if SAVE_NORMALIZATION_FILE:
#     scaler = normalize_and_save(X,time_flag=True)
#     X = normalize_std_scaler(X, scaler)
# else:
#     X = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')

# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)

# environments = list(zip(X_tensor, y_tensor))

class IRMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, output_dim=1):
        super().__init__()
        # 표현 학습망
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # 회귀 헤드 (bias=False 로 scale penalty 적용)
        self.head = nn.Linear(256, output_dim, bias=False)

    def forward(self, x):
        z = self.phi(x)           # [batch, feature_dim]
        y_pred = self.head(z)     # [batch, output_dim]
        return y_pred, z
    
def irm_regression_step(model, environments, λ, optimizer, device):
    """
    environments: list of (x_e, y_e) torch.Tensor, y_e shape [batch, output_dim]
    λ: IRM penalty weight
    """
    model.train()
    optimizer.zero_grad()

    mse = nn.MSELoss(reduction='mean')
    env_losses, penalties = [], []

    for x_e, y_e in environments:
        x_e, y_e = x_e.to(device), y_e.to(device)
        y_pred, _ = model(x_e)

        # 1) 환경별 MSE
        loss_e = mse(y_pred, y_e)
        env_losses.append(loss_e)

        # 2) IRM 페널티: scale 파라미터에 대한 그래디언트 제곱
        scale = torch.tensor(1.0, requires_grad=True, device=device)
        scaled_pred = y_pred * scale
        loss_scaled = mse(scaled_pred, y_e)
        grad_scale = torch.autograd.grad(loss_scaled, [scale], create_graph=True)[0]
        penalties.append(grad_scale.pow(2))

    # 전체 목적함수
    loss = torch.stack(env_losses).mean() + λ * torch.stack(penalties).mean()
    loss.backward()
    optimizer.step()

    return loss.item()

# 하이퍼파라미터 예시
input_dim    = len(p.feature_list)     # 입력 특성 개수
hidden_dim   = 256
feature_dim  = 64
output_dim   = len(p.output_list)       # 예측 대상 scalar
λ            = 5e2
lr           = 1e-3
num_epochs   = 100000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = IRMRegressor(input_dim, hidden_dim, feature_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if __name__ == "__main__":
    for epoch in range(1, num_epochs+1):
        loss = irm_regression_step(model, environments, λ, optimizer, device)
        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] IRM-reg loss = {loss:.4f}")

    model_name = name_time('DNN_DAB_est.pth')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model, model_path)
    print(f"Saved model to {model_path}")