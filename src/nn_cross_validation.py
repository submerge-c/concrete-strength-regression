import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


# ============================================================
# 1. 固定随机种子
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. 读取数据
# ============================================================

df = pd.read_excel("data/Concrete_Data.xls")

df.columns = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength",
]


X = df.drop(columns=["strength"])
y = df["strength"]


# 与传统机器学习保持完全相同的 80/20 划分
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 神经网络交叉验证只使用训练集
X = X_train.values
y = y_train.values


# ============================================================
# 3. GPU / CPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# ============================================================
# 4. 神经网络
# ============================================================

class ConcreteNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 5. 5折交叉验证
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


r2_scores = []
mae_scores = []
rmse_scores = []


# ============================================================
# 6. 每一折分别训练一个新的神经网络
# ============================================================

for fold, (train_index, val_index) in enumerate(
    kf.split(X),
    start=1,
):

    print(f"\n==============================")
    print(f"Fold {fold}")
    print("==============================")


    # --------------------------------------------------------
    # 当前折的数据
    # --------------------------------------------------------

    X_train = X[train_index]
    X_val = X[val_index]

    y_train = y[train_index]
    y_val = y[val_index]


    # --------------------------------------------------------
    # 标准化
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device,
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)


    # --------------------------------------------------------
    # 每一折都必须重新创建一个模型
    # --------------------------------------------------------

    torch.manual_seed(SEED + fold)

    model = ConcreteNet().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    epochs = 5000
    patience = 300

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    patience_counter = 0


    for epoch in range(epochs):

        # 训练
        model.train()

        train_pred = model(X_train_tensor)

        train_loss = criterion(
            train_pred,
            y_train_tensor,
        )

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()


        # 验证
        model.eval()

        with torch.no_grad():

            val_pred = model(X_val_tensor)

            val_loss = criterion(
                val_pred,
                y_val_tensor,
            )


        # 保存最佳模型
        if val_loss.item() < best_val_loss:

            best_val_loss = val_loss.item()

            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= patience:
            break


    # --------------------------------------------------------
    # 恢复当前折最佳模型
    # --------------------------------------------------------

    model.load_state_dict(best_model_state)

    model.eval()


    # --------------------------------------------------------
    # 当前折最终预测
    # --------------------------------------------------------

    with torch.no_grad():

        val_pred = (
            model(X_val_tensor)
            .cpu()
            .numpy()
            .flatten()
        )


    # --------------------------------------------------------
    # 当前折指标
    # --------------------------------------------------------

    r2 = r2_score(
        y_val,
        val_pred,
    )

    mae = mean_absolute_error(
        y_val,
        val_pred,
    )

    rmse = root_mean_squared_error(
        y_val,
        val_pred,
    )


    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)


    print(f"Best Epoch = {best_epoch}")
    print(f"R²         = {r2:.3f}")
    print(f"MAE        = {mae:.3f} MPa")
    print(f"RMSE       = {rmse:.3f} MPa")


# ============================================================
# 7. 计算5折平均结果
# ============================================================

print("\n================================")
print("5折交叉验证最终结果")
print("================================")

print(
    f"平均 R²   = {np.mean(r2_scores):.3f}"
)

print(
    f"R² 标准差 = {np.std(r2_scores):.3f}"
)

print(
    f"平均 MAE  = {np.mean(mae_scores):.3f} MPa"
)

print(
    f"平均 RMSE = {np.mean(rmse_scores):.3f} MPa"
)

print("\n每折 R²：")

for i, score in enumerate(
    r2_scores,
    start=1,
):
    print(
        f"Fold {i}: {score:.3f}"
    )
    