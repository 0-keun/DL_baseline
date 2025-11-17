# train_transformer_adv.py
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import amp
import numpy as np

# 사용자 유틸 (원본 경로 유지)
from utils.data_processing import normalize_and_save, add_normal_class, read_all_csv_to_np_list, make_sequence_dataset, load_and_normalize, normalize_std_scaler
from utils.utils import load_json, name_to_dir, add_dir_end, name_time

# ---------------------------
# 설정 불러오기
# ---------------------------
p = load_json(file_name='./params.json')
MODEL_DIR = name_to_dir(name='model', time_flag=True)
SAVE_NORMALIZATION_FILE = False

# 하이퍼파라미터
BATCH_SIZE = p.batch_size if hasattr(p, 'batch_size') else 64
SEQ_LEN = p.time_steps if hasattr(p, 'time_steps') else 250
FEATURE_DIM = len(p.feature_list) if hasattr(p, 'feature_list') else 3
NUM_CLASSES = 3 # len(p.classes_list) + 1  # add_normal_class 사용
EPOCH = 1000

# Adversarial training config
ADV_TRAIN = True
ADV_METHOD = "pgd"   # "fgsm" or "pgd"
ADV_EPS = 2e-2       # L_inf epsilon (입력이 표준화된 경우 작게 설정)
ADV_ALPHA = 5e-3     # PGD step size
ADV_STEPS = 3        # PGD steps
ADV_LAMBDA = 0.5     # adversarial loss 비중

# dir, file name
MODEL_DIR = name_to_dir(name='model', time_flag=True)
MODEL_TEMP_DIR = add_dir_end(MODEL_DIR, 'temp/')

CONTINUE_TRAIN = False  # 이어서 학습 여부
LOAD_MODEL_PATH = "./model/model_251024/transformer_182326.pth"  # 불러올 모델 경로 수정


# ---------------------------
# Positional Encoding
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---------------------------
# Transformer Classifier
# ---------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, feature_dim=3, seq_len=250, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder(x)
        cls_output = x[:, 0, :]
        return self.fc_out(cls_output)

# ---------------------------
# Adversarial attack helpers
# ---------------------------
def fgsm_attack(model, x, y, eps, criterion, device):
    # x: tensor (B, T, F), requires grad
    x_adv = x.detach().clone().requires_grad_(True)
    # generate logits/loss in float (disable autocast for stability)
    with amp.autocast('cuda', enabled=False):
        logits = model(x_adv)
        loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = x_adv + eps * grad.sign()
    return x_adv.detach()

def pgd_attack(model, x, y, eps, alpha, steps, criterion, device, clamp=None):
    x0 = x.detach()
    x_adv = x0.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with amp.autocast('cuda', enabled=False):
            logits = model(x_adv)
            loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project to L_inf ball around x0
        delta = torch.clamp(x_adv - x0, min=-eps, max=eps)
        x_adv = (x0 + delta).detach()
        if clamp is not None:
            lo, hi = clamp
            x_adv = x_adv.clamp(lo, hi)
    return x_adv

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    # 데이터 준비
    X_input, y_output = make_sequence_dataset(p.train_data_dir, p.time_steps, p.feature_list, p.classes_list)

    if SAVE_NORMALIZATION_FILE:
        features_data, _ = read_all_csv_to_np_list('./dataset/dataset_normal_251023', p.feature_list, p.classes_list, dim_reduction=True)
        scaler = normalize_and_save(np.squeeze(features_data), time_flag=True)
        X_input = normalize_std_scaler(X_input, scaler)
    else:
        X_input = load_and_normalize(X_input, './scaler/scaler_251023/mean_152218.npy', './scaler/scaler_251023/scale_152218.npy')

    y_output = add_normal_class(y_output)

    # 클래스 분포 출력 (디버깅)
    class_counts = [0] * (NUM_CLASSES)
    for out in y_output:
        for i, v in enumerate(out):
            class_counts[i] += int(v)
    print("sample distribution by class:", class_counts)

    # numpy -> tensor (X float, y long)
    X_input = torch.from_numpy(X_input).float()
    y_output = np.argmax(y_output, axis=1)
    y_output = torch.from_numpy(y_output).long()  # 중요: CrossEntropyLoss용 long
    print(y_output)

    # 새 텐서 생성 (기본값 0)
    mapped = torch.zeros_like(y_output)
    mapped[(y_output >= 4) & (y_output <= 7)] = 1
    mapped[y_output == 8] = 3
    print(mapped)

    y_output = mapped

    # data limits (for clamping adversarial examples)
    data_min = float(torch.min(X_input))
    data_max = float(torch.max(X_input))

    train_loader = DataLoader(TensorDataset(X_input, y_output), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier(FEATURE_DIM, SEQ_LEN, d_model=64, nhead=4, num_layers=2, num_classes=NUM_CLASSES).to(device)



    if CONTINUE_TRAIN and os.path.exists(LOAD_MODEL_PATH):
        checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded pretrained weights from: {LOAD_MODEL_PATH}")
    else:
        print("⚠️ No pretrained model loaded (training from scratch)")

    # optionally add class weights for imbalance (uncomment if desired)
    # counts = torch.tensor(class_counts, dtype=torch.float)
    # weights = (1.0 / (counts + 1e-9))
    # weights = weights / weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # AMP scaler
    scaler = amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    INPUT_CLAMP = (data_min, data_max)

    # training loop with NaN/Inf guard and adversarial

    print(f"Device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    stop_flag = False
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            # natural (clean) loss (with amp)
            with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs_nat = model(inputs)
                loss_nat = criterion(outputs_nat, labels)

            # adversarial example generation (use autocast disabled for stability)
            if ADV_TRAIN:
                if ADV_METHOD.lower() == "fgsm":
                    x_adv = fgsm_attack(model, inputs, labels, eps=ADV_EPS, criterion=criterion, device=device)
                else:
                    x_adv = pgd_attack(model, inputs, labels, eps=ADV_EPS, alpha=ADV_ALPHA,
                                       steps=ADV_STEPS, criterion=criterion, device=device, clamp=INPUT_CLAMP)
                # compute adversarial loss (we allow AMP here but attacks were computed with autocast disabled)
                with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs_adv = model(x_adv)
                    loss_adv = criterion(outputs_adv, labels)

                loss = (1.0 - ADV_LAMBDA) * loss_nat + ADV_LAMBDA * loss_adv
            else:
                loss = loss_nat

            # NaN/Inf guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at epoch {epoch+1}, batch {batch_idx}. Stopping training.")
                stop_flag = True
                break

            # optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[{epoch+1}/{EPOCH}] Avg Loss: {avg_loss:.8f}")

        # save if valid
        if stop_flag:
            print("Stopping early due to invalid loss.")
            break
        else:
            if epoch % 10 == 0:
                temp_filename = f"transformer_3F_epoch{epoch+1}.pth"
                save_path = os.path.join(MODEL_TEMP_DIR, name_time(temp_filename))
                torch.save(model.state_dict(), save_path)

    # final save
    final_path = os.path.join(MODEL_DIR, name_time("transformer_3F.pth"))
    torch.save(model.state_dict(), final_path)
    print("Training finished. Model saved to:", final_path)
