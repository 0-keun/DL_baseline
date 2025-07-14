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
from gpytorch.models.deep_gps import DeepGPLayer,DeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel

TRAIN = True
TEST = True
USE_DNGPA = True
LATENT_DIM = 16
NUM_INDUCING = 64
NUM_EPOCHS = 1000
LR = 0.01


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=NUM_INDUCING):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DeepGPModel(DeepGP):
    def __init__(self, input_dims):
        hidden_layer = ToyDeepGPHiddenLayer(input_dims=input_dims, output_dims=LATENT_DIM)
        last_layer = ToyDeepGPHiddenLayer(input_dims=LATENT_DIM, output_dims=1)
        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        # Sample from hidden GP layer
        hidden_dist = self.hidden_layer(x)
        hidden_sample = hidden_dist.rsample(torch.Size([1]))[0]
        # Output layer
        return self.last_layer(hidden_sample)

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_dist = self.forward(x)
            return self.likelihood(latent_dist).mean

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=255, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
    print(f"   MRE: {sum(e_list)/len(e_list):.2f}%")

def plot_predictions(y_true, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(y_pred.shape[1]):
        plt.figure(figsize=(8, 4))
        plt.plot(y_true[:, i], label='Ground Truth')
        plt.plot(y_pred[:, i], label='Prediction')
        plt.title(f"Output {i}: Prediction vs Ground Truth")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"output_{i}_prediction.png"))
        plt.close()

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

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    if USE_DNGPA:
        svd = TruncatedSVD(n_components=LATENT_DIM).fit(y)
        y_svd = svd.transform(y)
        joblib.dump(svd, os.path.join(save_dir, "svd.pkl"))
    else:
        y_svd = y

    for i in range(y_svd.shape[1]):
        print(f"\n=== Training Latent Output {i} ===")
        y = y_svd[:, i]
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1)

        model = DeepGPModel(input_dims=X_tensor.shape[1])
        likelihood = model.likelihood
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))

        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"model_layer_{i}.pth")
        torch.save(likelihood.state_dict(), f"likelihood_layer_{i}.pth")

    # for i in range(y_svd.shape[1]):
    #     print(f"\n=== Training Latent Output {i} ===")
    #     y_tensor = torch.tensor(y_svd[:, i], dtype=torch.float32)

    #     feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1])
    #     num_inducing = 64
    #     torch.manual_seed(42)
    #     rand_indices = torch.randperm(X_tensor.size(0))[:num_inducing]
    #     inducing_points = X_tensor[rand_indices]
        
    #     with gpytorch.settings.cholesky_jitter(1e-4):
    #         model = DKLGPModel(feature_extractor, inducing_points)
    #         likelihood = gpytorch.likelihoods.GaussianLikelihood()

    #         model.train()
    #         likelihood.train()

    #     optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)
    #     mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))

    #     for epoch in range(10000):
    #         optimizer.zero_grad()
    #         output = model(X_tensor)
    #         loss = -mll(output, y_tensor)
    #         loss.backward()
    #         optimizer.step()

    #     torch.save(model.state_dict(), f"{save_dir}/model_{i}.pth")
    #     torch.save(likelihood.state_dict(), f"{save_dir}/likelihood_{i}.pth")
    #     torch.save(feature_extractor.state_dict(), f"{save_dir}/feature_{i}.pth")

if TEST:
    p = load_json('./params.json')
    df_test = pd.read_csv(p.test_data_dir)
    X_new = df_test[p.feature_list].values
    y_true = df_test[p.output_list].values

    model_dir = os.path.join("./model/dkl_model")
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    svd = joblib.load(os.path.join(model_dir, "svd.pkl")) if USE_DNGPA else None

    z_preds = []
    dim = LATENT_DIM if USE_DNGPA else y_true.shape[1]

    for i in range(dim):
        print(f"Predicting Latent Output {i}")

        feature_extractor = FeatureExtractor(input_dim=X_tensor.shape[1])
        feature_extractor.load_state_dict(torch.load(f"{model_dir}/feature_{i}.pth"))
        inducing_points = X_tensor[:64]

        model = DKLGPModel(feature_extractor, inducing_points)
        model.load_state_dict(torch.load(f"{model_dir}/model_{i}.pth"))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.load_state_dict(torch.load(f"{model_dir}/likelihood_{i}.pth"))

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(X_tensor))
            z_preds.append(preds.mean.numpy())

    z_pred = np.stack(z_preds, axis=1)
    y_pred = svd.inverse_transform(z_pred) if USE_DNGPA else z_pred

    evaluate_prediction(y_true, y_pred)
    plot_predictions(y_true, y_pred)
    pd.DataFrame(y_pred, columns=[f"y{i}_pred" for i in range(y_pred.shape[1])]).to_csv("predicted.csv", index=False)