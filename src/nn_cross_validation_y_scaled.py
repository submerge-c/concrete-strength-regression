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


# ============================================================
# 3. 与传统机器学习保持相同的 80/20 划分
# ============================================================

X = df.drop(columns=["strength"])
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 交叉验证只使用这 80% 训练数据
X = X_train.values
y = y_train.values


# ============================================================
# 4. GPU / CPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# ============================================================
# 5. 定义神经网络
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
# 6. 5 折交叉验证
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
# 7. 每一折训练一个全新的神经网络
# ============================================================

for fold, (train_index, val_index) in enumerate(
    kf.split(X),
    start=1,
):

    print("\n==============================")
    print(f"Fold {fold}")
    print("==============================")


    # --------------------------------------------------------
    # 当前 Fold 的训练集和验证集
    # --------------------------------------------------------

    X_fold_train = X[train_index]
    X_fold_val = X[val_index]

    y_fold_train = y[train_index]
    y_fold_val = y[val_index]


    # --------------------------------------------------------
    # 标准化 X
    # --------------------------------------------------------

    x_scaler = StandardScaler()

    X_fold_train_scaled = x_scaler.fit_transform(
        X_fold_train
    )

    X_fold_val_scaled = x_scaler.transform(
        X_fold_val
    )


    # --------------------------------------------------------
    # 标准化 y
    # --------------------------------------------------------

    y_scaler = StandardScaler()

    y_fold_train_scaled = y_scaler.fit_transform(
        y_fold_train.reshape(-1, 1)
    )

    y_fold_val_scaled = y_scaler.transform(
        y_fold_val.reshape(-1, 1)
    )


    # --------------------------------------------------------
    # 转换成 Tensor
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_fold_train_scaled,
        dtype=torch.float32,
        device=device,
    )

    X_val_tensor = torch.tensor(
        X_fold_val_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_train_tensor = torch.tensor(
        y_fold_train_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_val_tensor = torch.tensor(
        y_fold_val_scaled,
        dtype=torch.float32,
        device=device,
    )


    # --------------------------------------------------------
    # 每一折重新创建模型
    # --------------------------------------------------------

    torch.manual_seed(SEED + fold)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold)

    model = ConcreteNet().to(device)


    # --------------------------------------------------------
    # 损失函数和优化器
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    for epoch in range(epochs):

        # ======================
        # 训练
        # ======================

        model.train()

        train_pred = model(
            X_train_tensor
        )

        train_loss = criterion(
            train_pred,
            y_train_tensor,
        )

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()


        # ======================
        # 验证
        # ======================

        model.eval()

        with torch.no_grad():

            val_pred = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_pred,
                y_val_tensor,
            )


        # ======================
        # 保存最佳模型
        # ======================

        if val_loss.item() < best_val_loss:

            best_val_loss = val_loss.item()

            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        # ======================
        # Early Stopping
        # ======================

        if patience_counter >= patience:
            break


    # --------------------------------------------------------
    # 恢复当前 Fold 最佳模型
    # --------------------------------------------------------

    model.load_state_dict(
        best_model_state
    )

    model.eval()


    # --------------------------------------------------------
    # 最终预测
    # --------------------------------------------------------

    with torch.no_grad():

        val_pred_scaled = (
            model(X_val_tensor)
            .cpu()
            .numpy()
        )


    # --------------------------------------------------------
    # y 反标准化：恢复成 MPa
    # --------------------------------------------------------

    val_pred = y_scaler.inverse_transform(
        val_pred_scaled
    ).flatten()


    # --------------------------------------------------------
    # 当前 Fold 指标
    # --------------------------------------------------------

    r2 = r2_score(
        y_fold_val,
        val_pred,
    )

    mae = mean_absolute_error(
        y_fold_val,
        val_pred,
    )

    rmse = root_mean_squared_error(
        y_fold_val,
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
# 8. 五折平均结果
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