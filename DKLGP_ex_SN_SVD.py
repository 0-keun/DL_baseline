import math
import tqdm
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
import urllib.request
import os
from scipy.io import loadmat
from math import floor
import pandas as pd
from utils.utils import load_json, name_to_dir, plot_2lines_N, plot_std_N, plot_predictions
from utils.data_processing import evaluate_prediction
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from gpytorch.constraints import GreaterThan
from gpytorch.constraints import Interval

param = load_json('./params.json')

import os
import csv
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --------------------------
# SVD Training Function
# --------------------------
def train_SVD(y, n_components, save_path="./svd_dir/svd.pkl"):
    svd = TruncatedSVD(n_components=n_components).fit(y)
    joblib.dump(svd, os.path.join(save_path))
    
    return svd

def load_SVD(save_path):
    return joblib.load(save_path)

# --------------------------
# Data Preparation
# --------------------------

def get_svd_data(data, update_svd=False, latent_dim=2):
    svd = train_SVD(data, latent_dim, save_path=f"{SVD_DIR}/svd.pkl") if update_svd else load_SVD(f"{SVD_DIR}/svd.pkl")
    data_svd = svd.transform(data)
    return data_svd

def get_inv_svd_data(data_svd):
    svd = load_SVD(f"{SVD_DIR}/svd.pkl")
    data_inv = svd.inverse_transform(data_svd)
    return data_inv

def data_prepare(df, y_svd, idx, scaler_flag=False, svd_flag=False):
    ############ X ############
    X = df[param.feature_list].values
    if scaler_flag:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    else:
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        X_scaled = scaler.transform(X)

    X = torch.tensor(X_scaled, dtype=torch.float32)
    x_tensor = X[:, :].contiguous()

    ############ y ############
    if svd_flag:
        y = y_svd[:, idx].reshape(-1, 1)
    else:
        y = df[param.output_list[idx]].values.reshape(-1, 1)
    # print(f"y before scaling: \n {y}")
    if scaler_flag:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)
        joblib.dump(scaler, f"{MODEL_DIR}/scaler_y_{idx}.pkl")
    else:
        scaler = joblib.load(f"{MODEL_DIR}/scaler_y_{idx}.pkl")
        y_scaled = scaler.transform(y)

    y = torch.tensor(y_scaled, dtype=torch.float32)
    y_tensor = y[:].contiguous()
    y_tensor = y_tensor.squeeze(-1).contiguous()

    if torch.cuda.is_available():
        x_tensor, y_tensor = x_tensor.cuda(), y_tensor.cuda()
    
    return x_tensor, y_tensor

def inv_data(y, idx):
    y = y.reshape(-1,1)
    scaler = joblib.load(f"{MODEL_DIR}/scaler_y_{idx}.pkl")
    return scaler.inverse_transform(y).reshape(-1)

def inv_std(std, idx):
    scaler = joblib.load(f"{MODEL_DIR}/scaler_y_{idx}.pkl")

    return std * scaler.scale_[0]
# --------------------------
# Model Definition
# --------------------------

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super().__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, LATENT_DIM))

class LargeFeatureExtractorSN(torch.nn.Sequential):
    def __init__(self, data_dim, dropout_prob=0.1):
        super().__init__()
        self.add_module('linear1', spectral_norm(nn.Linear(data_dim, 1000)))
        self.add_module('relu1',   nn.ReLU())
        # self.add_module('dropout1', nn.Dropout(dropout_prob))
        
        self.add_module('linear2', spectral_norm(nn.Linear(1000, 500)))
        self.add_module('relu2',   nn.ReLU())
        # self.add_module('dropout2', nn.Dropout(dropout_prob))
        
        self.add_module('linear3', spectral_norm(nn.Linear(500, 50)))
        self.add_module('relu3',   nn.ReLU())
        # self.add_module('dropout3', nn.Dropout(dropout_prob))
        
        self.add_module('linear4', spectral_norm(nn.Linear(50, LATENT_DIM)))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, feature_extractor, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=LATENT_DIM)
        )
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        z = self.feature_extractor(x)
        # z = self.scale_to_bounds(z).clamp(-0.999, 0.999)
        mean_x  = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --------------------------
# Training
# --------------------------

def train_iter(training_iterations,optimizer,model,train_x,train_y,mll,likelihood):
    iterator = tqdm.tqdm(range(training_iterations))

    # CSV 로그 파일 열기
    log_file = open("train_log.csv", mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["iter", "loss", "noise", "lengthscale", "outputscale", "min_eigval"])

    for i in iterator:
        with gpytorch.settings.max_cholesky_size(2000), gpytorch.settings.cholesky_jitter(1e-3), gpytorch.settings.use_toeplitz(False):
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()
            if i % 2000 == 0:   # 매 10 step마다 기록
                with torch.no_grad():
                    # 주요 하이퍼파라미터
                    noise = likelihood.noise.item()
                    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().tolist()
                    outputscale = model.covar_module.outputscale.item()
                    grad = likelihood.noise_covar.raw_noise.grad
                    # 커널 행렬 고유값 최소 계산
                    K = model.covar_module(
                            model.scale_to_bounds(model.feature_extractor(train_x))
                        ).to_dense()
                    K_noise = K + noise * torch.eye(K.size(0), device=K.device)
                    eigvals = torch.linalg.eigvalsh(K_noise)
                    min_eigval = eigvals.min().item()

                    # 터미널 출력
                    print(f"[Iter {i}] loss={loss.item():.3f}, noise={noise:.2e}, "
                        f"lengthscale={lengthscale}, outputscale={outputscale:.2e}, "
                        f"min_eigval={min_eigval:.2e}, grad={grad}" )

            #         # CSV 저장
            #         log_writer.writerow([i, loss.item(), noise, lengthscale, outputscale, min_eigval])
            #         log_file.flush()
            # with torch.no_grad():
            #     K = model.covar_module(model.scale_to_bounds(model.feature_extractor(train_x))).to_dense()
            #     device = K.device
            #     eigvals = torch.linalg.eigvalsh(
            #         K + likelihood.noise * torch.eye(K.size(0), device=device)
            #     )
            #     print("Eigenvalues:", eigvals[:10], "...", eigvals[-10:])
            #     print("Min eigenvalue:", eigvals.min().item())
            #     print("Max eigenvalue:", eigvals.max().item())
            # if i % 50 == 0:  # 예: 50 step마다
            #     print("Iter", i)
            #     print("Noise:", likelihood.noise.item())
            #     print("Lengthscale:", model.covar_module.base_kernel.base_kernel.lengthscale.detach().cpu().numpy())
            #     print("Outputscale:", model.covar_module.base_kernel.outputscale.item())

def train(Epoch = 1000):
    df = pd.read_csv(param.train_data_dir)
    output_dim = len(param.output_list) if not SVD_FLAG else SVD_DIM
    if SVD_FLAG:
        y_svd = get_svd_data(df[param.output_list].values, update_svd=UPDATE_SVD, latent_dim=output_dim)
    else:
        y_svd = None
    for i in range(output_dim):
        train_x, train_y = data_prepare(df, y_svd, i, scaler_flag=SCALER_FLAG, svd_flag=SVD_FLAG)
        train_x, train_y = train_x.cuda(), train_y.cuda()

        # print(np.min(np.array(train_x.cpu())),np.max(np.array(train_x.cpu())))
        # print(np.min(np.array(train_y.cpu())),np.max(np.array(train_y.cpu())))

        data_dim = train_x.size(-1)
        feature_extractor = LargeFeatureExtractorSN(data_dim) if SN_FLAG else LargeFeatureExtractor(data_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(5e-3))
        model = GPRegressionModel(feature_extractor, train_x, train_y, likelihood)
        model.covar_module.base_kernel.register_constraint("raw_lengthscale", Interval(1e-2, 4))
        model.covar_module.register_constraint("raw_outputscale", Interval(1e-4, 1e1))
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=0.001)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        train_iter(Epoch,optimizer,model,train_x,train_y,mll,likelihood)

        model.eval()
        likelihood.eval()

        torch.save(model.state_dict(), f"{MODEL_DIR}/model_{i}.pth")
        torch.save(likelihood.state_dict(), f"{MODEL_DIR}/likelihood_{i}.pth")

# --------------------------
# Testing
# --------------------------

def test():
    df = pd.read_csv(param.test_data_dir)
    y_trues = []
    y_preds = []
    y_stds  = []
    output_dim = len(param.output_list) if not SVD_FLAG else SVD_DIM
    if SVD_FLAG:
        y_svd = get_svd_data(df[param.output_list].values, update_svd=False, latent_dim=output_dim)
        y_svd_ = get_svd_data(pd.read_csv(param.train_data_dir)[param.output_list].values, update_svd=False, latent_dim=output_dim)
    else:
        y_svd = None
        y_svd_ = None
    for i in range(output_dim):
        train_x, train_y = data_prepare(pd.read_csv(param.train_data_dir), y_svd_, i, scaler_flag=False, svd_flag=SVD_FLAG)
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = data_prepare(df, y_svd, i, scaler_flag=False, svd_flag=SVD_FLAG)
        test_x, test_y = test_x.cuda(), test_y.cuda()
        data_dim = test_x.size(-1)
        feature_extractor = LargeFeatureExtractorSN(data_dim) if SN_FLAG else LargeFeatureExtractor(data_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(feature_extractor, train_x, train_y, likelihood)

        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
        # print(f"test_y: {test_y}")
        model.load_state_dict(torch.load(f"{MODEL_DIR}/model_{i}.pth",map_location=torch.device("cuda"), weights_only=True))
        likelihood.load_state_dict(torch.load(f"{MODEL_DIR}/likelihood_{i}.pth",map_location=torch.device("cuda"), weights_only=True))

        model.eval()
        likelihood.eval()

        model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.use_toeplitz(False):    # preds = model(test_x)
            preds = likelihood(model(test_x))
            # print(f"preds: {preds}")
            mean = preds.mean
            var  = preds.variance

            # NaN 제거: mean은 0, var은 0 으로 대체
            mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
            var  = torch.where(torch.isnan(var),  torch.zeros_like(var),  var)

            # 분산 음수 클램핑 & std 계산
            var = var.clamp(min=0.0)
            std = var.sqrt()
        print(f"mean: {mean}")
        # print(f"var: {var}")
        y_pred = inv_data(mean.cpu().numpy().reshape(-1,1), i)
        y_std  = inv_std(std.cpu().numpy().reshape(-1,1), i)

        y_preds.append(y_pred)
        y_stds.append(y_std)


    if SVD_FLAG:
        y_preds = get_inv_svd_data(np.array(y_preds).T)
        Z_std_norm = np.array(y_stds).T

        scales = []
        for i in range(SVD_DIM):  # r
            scaler_i = joblib.load(f"{MODEL_DIR}/scaler_y_{i}.pkl")
            scales.append(scaler_i.scale_[0])

        scales = np.array(scales)  # (r,)

        # (4) 스케일러 역변환 (평균/분산 규칙)
        Z_var_raw = (Z_std_norm**2) # * (scales[None, :]**2)             # (n, r)

        # (5) SVD 역투영: 평균과 분산(대각) 계산
        svd = load_SVD(f"{SVD_DIR}/svd.pkl")
        V_r = svd.components_.T                  # (d, r)
        V_r_sq = V_r**2                          # (d, r)

        Y_var = Z_var_raw @ V_r_sq.T             # (n, d)  -> 각 출력 j의 분산
        Y_std = np.sqrt(Y_var)                   # (n, d)
        y_stds  = Y_std

    else:
        y_preds = np.array(y_preds).T
        y_stds = np.array(y_stds).T

    # Evaluate
    y_preds = np.array(y_preds)
    y_stds  = np.array(y_stds)
    y_trues = pd.read_csv(param.test_data_dir)[param.output_list].values

    print(len(y_preds))

    evaluate_prediction(y_trues, y_preds)
    # plot_predictions(y_trues, y_preds)

    y_stds = np.squeeze(y_stds)
    pred_df = pd.DataFrame(y_preds, columns=[f"y{i}_pred" for i in range(y_preds.shape[1])])
    std_df  = pd.DataFrame(y_stds,  columns=[f"y{i}_std"  for i in range(y_stds.shape[1])])

    # Add the function to create the directory.
    result_dir = name_to_dir(f"results/{SN}dklgp{SVD}")
    pred_df.to_csv(result_dir+"predicted.csv", index=False)
    std_df.to_csv(result_dir+"predicted_std.csv", index=False)

    mean_dir = name_to_dir(f"plots/{SN}dklgp{SVD}/mean")
    std_dir = name_to_dir(f"plots/{SN}dklgp{SVD}/std")
    mean_dir = name_to_dir(f"plots/{SN}dklgp{SVD}/mean")
    std_dir = name_to_dir(f"plots/{SN}dklgp{SVD}/std")
    n_out = y_preds.shape[1]
    plot_2lines_N(y_trues, y_preds, n_out, mean_dir)
    plot_std_N(y_stds, n_out, std_dir)

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    TRAIN = False
    TEST  = True
    LATENT_DIM = 4
    EPOCH = 14000

    SN_FLAG = True
    SVD_FLAG = True
    UPDATE_SVD = True
    SCALER_FLAG = True
    SVD_DIM = 4 # len(param.output_list)
    SN = "sn_" if SN_FLAG else ""
    SVD = "_svd" if SVD_FLAG else ""

    MODEL_DIR = name_to_dir(f"model/{SN}dkl{SVD}/")
    SVD_DIR = name_to_dir(f"model/{SN}dkl{SVD}/svd_dir/")
    # MODEL_DIR = name_to_dir(f"model_temp_best/{SN}dkl{SVD}/")
    # SVD_DIR = name_to_dir(f"model_temp_best/{SN}dkl{SVD}/svd_dir/")

    if TRAIN:
        train(Epoch=EPOCH)
    if TEST:
        test()