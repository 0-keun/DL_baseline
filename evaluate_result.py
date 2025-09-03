import numpy as np

def _as_2d(a):
    """(n_samples, n_outputs) 형태로 강제."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    return a

def rmse(y_true, y_pred):
    """
    반환: per-output RMSE (shape: [d]), mean_RMSE (float)
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    err = y_pred - y_true
    per_dim = np.sqrt(np.mean(err**2, axis=0))
    return per_dim, float(per_dim.mean())

def mae(y_true, y_pred):
    """
    반환: per-output MAE (shape: [d]), mean_MAE (float)
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    err = np.abs(y_pred - y_true)
    per_dim = np.mean(err, axis=0)
    return per_dim, float(per_dim.mean())

def mape(y_true, y_pred, eps=1e-8):
    """
    반환: per-output MAPE (%) (shape: [d]), mean_MAPE (%) (float)
    - y_true가 0(또는 eps 이하)인 항목은 분모 문제로 제외.
    - 해당 차원에서 전부 제외되면 np.nan 반환.
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    abs_true = np.abs(y_true)
    mask = abs_true > eps  # 유효 항목만
    per_dim = np.empty(y_true.shape[1], dtype=float)
    per_dim.fill(np.nan)
    for j in range(y_true.shape[1]):
        m = mask[:, j]
        if np.any(m):
            per_dim[j] = 100.0 * np.mean(np.abs((y_pred[m, j] - y_true[m, j]) / y_true[m, j]))
        else:
            per_dim[j] = np.nan  # 전부 0이면 정의 불가
    return per_dim, float(np.nanmean(per_dim))

def r2(y_true, y_pred):
    """
    반환: per-output R^2 (shape: [d]), mean_R2 (float)
    - y_true가 상수(분산 0)인 차원은:
        - 완벽히 예측 시 1.0
        - 아니면 0.0  (sklearn의 관례와 유사)
    """
    y_true, y_pred = _as_2d(y_true), _as_2d(y_pred)
    y_mean = np.mean(y_true, axis=0)
    ss_res = np.sum((y_pred - y_true)**2, axis=0)
    ss_tot = np.sum((y_true - y_mean)**2, axis=0)

    per_dim = np.empty(y_true.shape[1], dtype=float)
    for j in range(y_true.shape[1]):
        if np.isclose(ss_tot[j], 0.0):
            # 상수 타겟: 완벽 예측이면 1.0, 아니면 0.0
            per_dim[j] = 1.0 if np.allclose(y_pred[:, j], y_true[:, j]) else 0.0
        else:
            per_dim[j] = 1.0 - (ss_res[j] / ss_tot[j])
    return per_dim, float(per_dim.mean())

def summarize_metrics(y_true, y_pred):
    """
    네 가지 지표를 한 번에 요약 출력.
    반환: dict
    """
    rmse_dim, rmse_mean = rmse(y_true, y_pred)
    mae_dim, mae_mean   = mae(y_true, y_pred)
    mape_dim, mape_mean = mape(y_true, y_pred)
    r2_dim, r2_mean     = r2(y_true, y_pred)

    return {
        "RMSE_per_output": rmse_dim,
        "RMSE_mean": rmse_mean,
        "MAE_per_output": mae_dim,
        "MAE_mean": mae_mean,
        "MAPE_per_output(%)": mape_dim,
        "MAPE_mean(%)": mape_mean,
        "R2_per_output": r2_dim,
        "R2_mean": r2_mean,
    }

# ===== 사용 예시 =====
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 100, 16
    y_true = rng.normal(size=(n, d))
    y_pred = y_true + 0.1 * rng.normal(size=(n, d))  # 약간의 잡음

    summary = summarize_metrics(y_true, y_pred)
    for k, v in summary.items():
        print(k, ":", v)
