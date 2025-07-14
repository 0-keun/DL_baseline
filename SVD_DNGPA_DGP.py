import os
import torch
import gpytorch
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from gpytorch.models.deep_gps import DeepGPLayer,DeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from utils.utils import load_json

# -------------------------------
# Configuration
# -------------------------------
TRAIN = True
TEST = True
USE_SVD = True
LATENT_DIM = 16
NUM_INDUCING = 64
NUM_EPOCHS = 1000
LR = 0.01

# -------------------------------
# Model Definition
# -------------------------------
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
        hidden_sample = hidden_dist.rsample()
        # Output layer
        return self.last_layer(hidden_sample)

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_dist = self.forward(x)
            return self.likelihood(latent_dist).mean

# -------------------------------
# Evaluation Utils
# -------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mre = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.4f}, MRE: {mre:.2f}%, R2: {r2:.4f}")

def plot_results(y_true, y_pred, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction.png")
    plt.close()

# -------------------------------
# Training
# -------------------------------
if TRAIN:
    p = load_json("params.json")
    df = pd.read_csv(p.train_data_dir)
    X = df[p.feature_list].values
    Y = df[p.output_list].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    joblib.dump(scaler, "scaler.pkl")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    if USE_SVD:
        svd = TruncatedSVD(n_components=LATENT_DIM).fit(Y)
        Y_latent = svd.transform(Y)
        joblib.dump(svd, "svd.pkl")
    else:
        Y_latent = Y

    for i in range(Y_latent.shape[1]):
        y = Y_latent[:, i]
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1)

        model = DeepGPModel(input_dims=X_tensor.shape[1])
        likelihood = model.likelihood
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))
        print("Output mean shape:", output.mean.shape)
        print("Target shape:", y_tensor.shape)
        for epoch in range(NUM_EPOCHS):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"model_layer_{i}.pth")
        torch.save(likelihood.state_dict(), f"likelihood_layer_{i}.pth")

# -------------------------------
# Testing
# -------------------------------
if TEST:
    p = load_json("params.json")
    df_test = pd.read_csv(p.test_data_dir)
    X_test = df_test[p.feature_list].values
    Y_true = df_test[p.output_list].values

    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    if USE_SVD:
        svd = joblib.load("svd.pkl")

    preds = []
    for i in range(Y_true.shape[1] if not USE_SVD else LATENT_DIM):
        model = DeepGPModel(input_dims=X_tensor.shape[1])
        likelihood = model.likelihood
        model.load_state_dict(torch.load(f"model_layer_{i}.pth"))
        likelihood.load_state_dict(torch.load(f"likelihood_layer_{i}.pth"))
        y_pred = model.predict(X_tensor).numpy()
        preds.append(y_pred)

    Z_pred = np.stack(preds, axis=1)
    Y_pred = svd.inverse_transform(Z_pred) if USE_SVD else Z_pred

    # Evaluate first output
    evaluate(Y_true[:, 0], Y_pred[:, 0])
    plot_results(Y_true[:, 0], Y_pred[:, 0])
