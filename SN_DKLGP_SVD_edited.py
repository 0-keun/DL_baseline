import os
import torch
import gpytorch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from utils.utils import load_json, plot_predictions, name_to_dir, plot_2lines_N, plot_std_N
from utils.data_processing import load_and_normalize, evaluate_prediction
from utils.models import FeatureExtractor, FeatureExtractorWithoutSN, DKLGPModel
import time

param = load_json('./params.json')
TRAIN = True
TEST = True
USE_SN  = True
USE_SVD = True
LATENT_DIM = len(param.output_list)
NUM_INDUCING = 64
EPOCH = 10000

UPDATE_SVD = False

model_list = [USE_SN, True, USE_SVD]
mname_list = ['sn_', 'dkl', '_svd']
for i in range(3):
    if not model_list[i]:
        mname_list[i] = ''

model_name = mname_list[0]+mname_list[1]+mname_list[2]+'_model'

SVD_DIR = name_to_dir("svd_dir")
SVD_PATH = os.path.join(SVD_DIR, "svd.pkl")
MODEL_DIR = name_to_dir(model_name, time_flag=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
SCALER_M_PATH = './scaler/scaler_250904/mean_152528.npy'
SCALER_S_PATH = './scaler/scaler_250904/scale_152528.npy'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# SVD Training Function
# --------------------------
def train_SVD(y, n_components, save_path="./svd_dir/svd.pkl"):
    svd = TruncatedSVD(n_components=n_components).fit(y)
    joblib.dump(svd, os.path.join(save_path))
    
    return svd

def load_SVD(path):
    return joblib.load(path)


# --------------------------
# Training Phase            
# --------------------------
def train_and_save_model(save_dir, X_tensor, y_svd, i):
    print(f"\n=== Training Latent Output {i} ===")
    y_tensor = torch.tensor(y_svd[:, i], dtype=torch.float32).to(device)

    feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1]).to(device) if USE_SN else FeatureExtractorWithoutSN(input_dim=X_tensor.shape[1]).to(device)
    rand_indices = torch.randperm(X_tensor.size(0))[:NUM_INDUCING].to(device)
    inducing_points = X_tensor[rand_indices].to(device)

    model = DKLGPModel(feature_extractor, inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))

    for _ in range(EPOCH):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = -mll(output, y_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"{save_dir}/model_{i}.pth")
    torch.save(likelihood.state_dict(), f"{save_dir}/likelihood_{i}.pth")
    torch.save(feature_extractor.state_dict(), f"{save_dir}/feature_{i}.pth")
    torch.save(inducing_points.cpu(), f"{save_dir}/inducing_{i}.pt")

# --------------------------
# Test Phase
# --------------------------
def test_model(X_tensor, i):
    print(f"Predicting Latent Output {i}")

    feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1]).to(device) if USE_SN else FeatureExtractorWithoutSN(input_dim=X_tensor.shape[1]).to(device)
    feature_extractor.load_state_dict(torch.load(f"{MODEL_DIR}/feature_{i}.pth", map_location=device))
    inducing_points = torch.load(f"{MODEL_DIR}/inducing_{i}.pt").to(device)

    model = DKLGPModel(feature_extractor, inducing_points).to(device)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/model_{i}.pth", map_location=device))
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.load_state_dict(torch.load(f"{MODEL_DIR}/likelihood_{i}.pth", map_location=device))

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # 정확한 분산 계산 모드로 전환 (fast_pred_var 끄기)
        with gpytorch.settings.fast_pred_var(False):
            preds = likelihood(model(X_tensor))

        # mean, var 꺼내오기
        mean = preds.mean
        var  = preds.variance

        # NaN 제거: mean은 0, var은 0 으로 대체
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        var  = torch.where(torch.isnan(var),  torch.zeros_like(var),  var)

        # 분산 음수 클램핑 & std 계산
        var = var.clamp(min=0.0)
        std = var.sqrt()

        z_means.append(mean.cpu().numpy())
        z_stds.append(std.cpu().numpy())
    
    return z_means, z_stds


if TRAIN:
    df = pd.read_csv(param.train_data_dir)

    X = df[param.feature_list].values
    y = df[param.output_list].values

    X_scaled = load_and_normalize(X,SCALER_M_PATH,SCALER_S_PATH)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    if USE_SVD:
        if UPDATE_SVD:
            svd = train_SVD(y, LATENT_DIM, save_path=SVD_PATH)
            y_svd = svd.transform(y)
        else:
            svd = load_SVD(SVD_PATH)

        y_svd = svd.transform(y)
    else:
        y_svd = y

    for i in range(y_svd.shape[1]):
        train_and_save_model(MODEL_DIR, X_tensor, y_svd, i)

# --------------------------
# Test Phase
# --------------------------
if TEST:
    df_test = pd.read_csv(param.test_data_dir)
    X_new = df_test[param.feature_list].values
    y_true = df_test[param.output_list].values

    X_scaled = load_and_normalize(X_new,SCALER_M_PATH,SCALER_S_PATH)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    svd = load_SVD(SVD_PATH) if USE_SVD else None
    dim = LATENT_DIM if USE_SVD else y_true.shape[1]

    z_means = []
    z_stds = []

    for i in range(dim):
        z_means, z_stds = test_model(X_tensor, i)

    z_mean = np.stack(z_means, axis=1)
    z_std = np.stack(z_stds, axis=1)

    y_pred = svd.inverse_transform(z_mean) if USE_SVD else z_mean
    y_std = svd.inverse_transform(z_std) if USE_SVD else z_std
    
    # Evaluate
    evaluate_prediction(y_true, y_pred)
    plot_predictions(y_true, y_pred)

    pred_df = pd.DataFrame(y_pred, columns=[f"y{i}_pred" for i in range(y_pred.shape[1])])
    std_df  = pd.DataFrame(y_std,  columns=[f"y{i}_std"  for i in range(y_std.shape[1])])
    pred_df.to_csv("predicted.csv", index=False)
    std_df.to_csv("predicted_std.csv", index=False)

    mean_dir = name_to_dir("plots_mean")
    std_dir = name_to_dir("plots_std")
    n_out = y_pred.shape[1]
    plot_2lines_N(y_true, y_pred, n_out, mean_dir)
    plot_std_N(y_std, n_out, std_dir)