import os
import torch
import torch.nn as nn
import gpytorch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from utils.utils import load_json
from torch.nn.utils import spectral_norm

p = load_json('./params.json')
TRAIN = False
TEST = True
USE_DNGPA = True
LATENT_DIM = len(p.output_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=255, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(


            spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, output_dim))
        )

    def forward(self, x):
        return self.net(x)

class DKLGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, feature_extractor, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def evaluate_prediction(y_true, y_pred, epsilon=1e-8):
    e_list = []
    absolute_error_list = []
    for i in range(y_pred.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        mae = mean_absolute_error(y_t, y_p)
        relative_error = np.abs((y_p - y_t) / (y_t + epsilon))
        mre = np.mean(relative_error) * 100
        r2 = r2_score(y_t, y_p)

        print(f"\n>> [Output {i}]")
        print(f"   MAE: {mae:.4f}")
        print(f"   Mean Relative Error: {mre:.2f}%")
        print(f"   R² Score: {r2:.4f}")
        e_list.append(mre)
        absolute_error_list.append(mae)
    print(f"   MRE: {sum(e_list)/len(e_list):.2f}%")
    print(f"   MRE: {sum(absolute_error_list)/len(absolute_error_list):.4f}")

def plot_predictions(y_true, y_pred, y_std=None, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(y_pred.shape[1]):
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Prediction (Mean)')
        if y_std is not None:
            lower = y_pred[:, i] - 2 * y_std[:, i]
            upper = y_pred[:, i] + 2 * y_std[:, i]
            plt.fill_between(np.arange(len(y_pred)), lower, upper, alpha=0.3, label='±2σ Uncertainty')
        plt.title(f"Output {i}: Prediction with Uncertainty")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"output_{i}_prediction.png"))
        plt.close()

# --------------------------
# Training Phase
# --------------------------
if TRAIN:
    p = load_json('./params.json')
    df = pd.read_csv(p.train_data_dir)

    X = df[p.feature_list].values
    y = df[p.output_list].values

    save_dir = os.path.join("model", "dkl_model")
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    if USE_DNGPA:
        svd = TruncatedSVD(n_components=LATENT_DIM).fit(y)
        y_svd = svd.transform(y)
        joblib.dump(svd, os.path.join(save_dir, "svd.pkl"))
    else:
        y_svd = y

    for i in range(y_svd.shape[1]):
        print(f"\n=== Training Latent Output {i} ===")
        y_tensor = torch.tensor(y_svd[:, i], dtype=torch.float32).to(device)

        feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1]).to(device)
        num_inducing = 64
        torch.manual_seed(42)
        rand_indices = torch.randperm(X_tensor.size(0))[:num_inducing].to(device)
        inducing_points = X_tensor[rand_indices].to(device)

        model = DKLGPModel(feature_extractor, inducing_points).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))

        for epoch in range(10000):
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
if TEST:
    p = load_json('./params.json')
    df_test = pd.read_csv(p.test_data_dir)
    X_new = df_test[p.feature_list].values
    y_true = df_test[p.output_list].values

    model_dir = os.path.join("./model/dkl_model")
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    svd = joblib.load(os.path.join(model_dir, "svd.pkl")) if USE_DNGPA else None

    z_means = []
    z_stds = []
    dim = LATENT_DIM if USE_DNGPA else y_true.shape[1]

    for i in range(dim):
        print(f"Predicting Latent Output {i}")

        feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1]).to(device)
        feature_extractor.load_state_dict(torch.load(f"{model_dir}/feature_{i}.pth", map_location=device))
        inducing_points = torch.load(f"{model_dir}/inducing_{i}.pt").to(device)

        model = DKLGPModel(feature_extractor, inducing_points).to(device)
        model.load_state_dict(torch.load(f"{model_dir}/model_{i}.pth", map_location=device))
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        likelihood.load_state_dict(torch.load(f"{model_dir}/likelihood_{i}.pth", map_location=device))

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
            z_stds .append(std.cpu().numpy())

    z_mean = np.stack(z_means, axis=1)
    z_std = np.stack(z_stds, axis=1)

    # # NaN 포함된 샘플 출력
    # for i, row in enumerate(z_mean):
    #     if np.isnan(row).any():
    #         print(f"[NaN Detected @ Sample {i}] {row}")

    y_pred = svd.inverse_transform(z_mean) if USE_DNGPA else z_mean
    y_std = svd.inverse_transform(z_std) if USE_DNGPA else z_std

    evaluate_prediction(y_true, y_pred)
    plot_predictions(y_true, y_pred)

    # Save predictions & stddevs
    pred_df = pd.DataFrame(y_pred, columns=[f"y{i}_pred" for i in range(y_pred.shape[1])])
    std_df  = pd.DataFrame(y_std,  columns=[f"y{i}_std"  for i in range(y_std.shape[1])])
    pred_df.to_csv("predicted.csv", index=False)
    std_df.to_csv("predicted_std.csv", index=False)

    # Plot and save mean vs ground truth
    mean_dir = "plots_mean"; std_dir = "plots_std"
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(std_dir,  exist_ok=True)

    n_out = y_pred.shape[1]
    for i in range(n_out):
        plt.figure(figsize=(8,4))
        plt.plot(y_true[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Predicted Mean')
        plt.title(f'Output {i} Mean Prediction')
        plt.xlabel('Sample Index'); plt.ylabel('Value')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(mean_dir, f'output_{i}_mean.png'))
        plt.close()

    # Plot and save stddev
    for i in range(n_out):
        plt.figure(figsize=(8,4))
        plt.plot(y_std[:, i], label='Predicted Std Dev')
        plt.title(f'Output {i} Std Dev')
        plt.xlabel('Sample Index'); plt.ylabel('Std Dev')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(std_dir, f'output_{i}_std.png'))
        plt.close()
