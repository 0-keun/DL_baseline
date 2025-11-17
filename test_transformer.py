import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from train_transformer import TransformerClassifier
from utils.data_processing import load_serial_data_from_csv,make_sequence_dataset, add_normal_class, load_and_normalize
from utils.utils import get_confusion_mat

# ======================================================
# get params       
# ======================================================

import re
import json
from utils.utils import load_json, name_to_dir

p = load_json(file_name='./params.json')

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

NUM_CLASSES = 3

def get_params(filename):
    # 1) basename만 뽑아내고 싶으면 pathlib 사용
    from pathlib import Path
    stem = Path(filename).stem      # → 'LSTM_h10_layer3'

    # 2) 정규표현식 패턴
    pattern = r'LSTM_h(\d+)_layer(\d+)\_class(\d+)_(\d+).h5$'

    m = re.search(pattern, filename)
    if m:
        num1 = int(m.group(1))   # h 뒤 숫자
        num2 = int(m.group(2))   # layer 뒤 숫자
        num3 = int(m.group(3))
        return num1, num2, num3

# hidden_state, num_layer, _ = get_params(model_name)
X_input, y_output = make_sequence_dataset(p.test_data_dir,p.time_steps,p.feature_list,p.classes_list)
X_input = load_and_normalize(X_input,'./scaler/scaler_250610/mean_180723.npy','./scaler/scaler_250610/scale_180723.npy')
y_output = add_normal_class(y_output)

list = [0]*(NUM_CLASSES)
for output in y_output:
    list += output
print("sample distribution by class:", list)  

# ======================================================
# 환경 설정
# ======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================================================
# 1. 모델 불러오기
# ======================================================
model = TransformerClassifier(
    feature_dim=3, seq_len=250, d_model=64, nhead=4,
    num_layers=2, num_classes=NUM_CLASSES
).to(device)

# 저장된 가중치 로드
model.load_state_dict(torch.load("./model/model_251027/transformer_175308.pth", map_location=device, weights_only=True))
model.eval()

# ======================================================
# 2. 평가용 데이터 준비 (예: numpy → tensor 변환)
# ======================================================
# 예시: 이미 numpy로 저장된 테스트셋
# X_test = np.load("X_test.npy")   # shape: (N, 250, 3)
# y_test = np.load("y_test.npy")   # shape: (N,)

# numpy → tensor
X_test = torch.from_numpy(X_input).float()
y_test = torch.from_numpy(y_output).long()

# 새 텐서 생성 (기본값 0)
mapped = torch.zeros_like(y_test)
mapped[(y_test >= 4) & (y_test <= 7)] = 1
mapped[y_test == 8] = 3

y_test = mapped

# print(f"X_test: \n{X_test}")
print(f"X_test_N: \n{X_test.size(0)}")
print(f"X_test_T: \n{X_test.size(1)}")
print(f"X_test_D: \n{X_test.size(2)}")

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ======================================================
# 3. 평가 루프
# ======================================================
criterion = nn.CrossEntropyLoss()
total_loss = 0.0
correct = 0
total = 0

preds_list = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # print("outputs:", outputs.shape)

        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # print(f"preds:\n{preds}")
        # print(f"labels:\n{labels}")
        preds_list.extend(preds.tolist())

# print(preds_list)

avg_loss = total_loss / len(test_loader)
accuracy = correct / total * 100

if torch.is_tensor(y_test):
    y_test = y_test.detach().cpu().numpy()
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
if torch.is_tensor(preds_list):
    preds_list = preds_list.detach().cpu().numpy()
get_confusion_mat(y_test,preds_list,time_flag=True,save_csv=False)

print(f"✅ Test Loss: {avg_loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# # ======================================================
# # 4. 임의 샘플 예측
# # ======================================================
# sample = X_test[0].unsqueeze(0).to(device)  # (1, 250, 3)
# with torch.no_grad():
#     pred = model(sample)
#     label = torch.argmax(pred, dim=1).item()
#     label_map = {0: 'normal', 1: 'open', 2: 'short'}
#     print(f"🔍 Sample #0 Predicted Class → {label_map[label]}")
