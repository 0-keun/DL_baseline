import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from torch import amp 
import numpy as np
from utils.data_processing import normalize_and_save, add_normal_class, read_all_csv_to_np_list, make_sequence_dataset, load_and_normalize, normalize_std_scaler
# ======================================================
# get params       
# ======================================================

import json
from utils.utils import load_json, name_to_dir

p = load_json(file_name='./params_3F.json')

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

# ======================================================
# Positional Encoding
# ======================================================
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


# ======================================================
# Transformer Classifier
# ======================================================
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

if __name__ == '__main__':
    # ======================================================
    # Dummy Dataset (예시)
    # ======================================================
    batch_size = 32
    seq_len = 250
    feature_dim = 3
    num_classes = 3

    X_input, y_output = make_sequence_dataset(p.train_data_dir,p.time_steps,p.feature_list,p.classes_list)
    if SAVE_NORMALIZATION_FILE:
        features_data, _ = read_all_csv_to_np_list('./dataset/dataset_normal_250610',p.feature_list,p.classes_list,dim_reduction=True)
        scaler = normalize_and_save(np.squeeze(features_data),time_flag=True)
        X_input = normalize_std_scaler(X_input, scaler)
    else:
        X_input = load_and_normalize(X_input,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')
    y_output = add_normal_class(y_output)

    list = [0]*(len(p.classes_list)+1)
    for output in y_output:
        list += output
    print("sample distribution by class:", list)  

    X_input = torch.from_numpy(X_input).float()
    y_output = torch.from_numpy(y_output).float()

    # print(f"X_input: \n{X_input}")
    # print(f"y_output: \n{y_output}")

    # X = torch.randn(512, seq_len, feature_dim)
    # y = torch.randint(0, num_classes, (512,))
    train_loader = DataLoader(TensorDataset(X_input, y_output), batch_size=batch_size, shuffle=True)

    # ======================================================
    # 학습 세팅
    # ======================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier(feature_dim, seq_len, d_model=64, nhead=4, num_layers=2, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Triton 비사용: torch.compile 제거
    # model = torch.compile(model) 

    # 최신 AMP API 사용
    scaler = amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"[{epoch+1}/100] Loss: {total_loss/len(train_loader):.8f}")
        if not torch.isnan(loss):
            torch.save(model.state_dict(), "./model/transformer/model.pth")
        else:
            print(f"❌ NaN/Inf detected at epoch {epoch+1}. Training stopped.")
            break

    # ======================================================
    # 예측 예시
    # ======================================================
    # model.eval()
    # with torch.no_grad():
    #     sample = torch.randn(1, seq_len, feature_dim).to(device)
    #     pred = model(sample)
    #     label = torch.argmax(pred, dim=1).item()
    #     label_map = {0:'normal', 1:'open', 2:'short'}
    #     print(f"Predicted class → {label_map[label]}")
