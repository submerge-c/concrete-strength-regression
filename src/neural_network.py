import copy

import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


# ============================================================
# 1. 固定随机种子
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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
# 3. 划分输入 X 和目标 y
# ============================================================

X = df.drop(columns=["strength"])
y = df["strength"]


# ============================================================
# 4. 划分 Train / Validation / Test
# ============================================================

# 第一步：
# 留出 20% 作为最终测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 第二步：
# 从剩余 80% 中，再拿 20% 做验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2,
    random_state=42,
)

print("数据划分：")
print("训练集：", X_train.shape)
print("验证集：", X_val.shape)
print("测试集：", X_test.shape)


# ============================================================
# 5. 标准化
# ============================================================

scaler = StandardScaler()

# 只能用训练集拟合 StandardScaler
X_train_scaled = scaler.fit_transform(X_train)

# 验证集和测试集只能使用训练集得到的标准化参数
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. 转换成 PyTorch Tensor
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32,
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.values,
    dtype=torch.float32,
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32,
).reshape(-1, 1)


# ============================================================
# 7. 选择 GPU / CPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n使用设备：", device)


# ============================================================
# 8. 把 Tensor 移动到 GPU
# ============================================================

X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)

y_train_tensor = y_train_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# ============================================================
# 9. 定义神经网络
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
# 10. 创建模型
# ============================================================

model = ConcreteNet().to(device)

print("\n神经网络结构：")
print(model)


# ============================================================
# 11. 损失函数
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# 12. 优化器
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# ============================================================
# 13. Early Stopping 参数
# ============================================================

epochs = 5000

patience = 300

best_val_loss = float("inf")

best_model_state = None

best_epoch = 0

patience_counter = 0


# ============================================================
# 14. 开始训练
# ============================================================

print("\n开始训练：")

for epoch in range(epochs):

    # --------------------------------------------------------
    # 训练阶段
    # --------------------------------------------------------

    model.train()

    # 前向传播
    y_train_pred = model(X_train_tensor)

    # 计算训练损失
    train_loss = criterion(
        y_train_pred,
        y_train_tensor,
    )

    # 清除上一轮梯度
    optimizer.zero_grad()

    # 反向传播
    train_loss.backward()

    # 更新网络参数
    optimizer.step()


    # --------------------------------------------------------
    # 验证阶段
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        y_val_pred = model(X_val_tensor)

        val_loss = criterion(
            y_val_pred,
            y_val_tensor,
        )


    # --------------------------------------------------------
    # 保存验证集表现最好的模型
    # --------------------------------------------------------

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # 每100轮打印一次
    # --------------------------------------------------------

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f}"
        )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if patience_counter >= patience:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )

        break


# ============================================================
# 15. 恢复验证集表现最好的模型
# ============================================================

model.load_state_dict(best_model_state)

model = model.to(device)

print("\n最佳模型：")
print(f"Best Epoch    = {best_epoch}")
print(f"Best Val Loss = {best_val_loss:.4f}")


# ============================================================
# 16. 最终预测
# ============================================================

model.eval()

with torch.no_grad():

    train_pred_tensor = model(X_train_tensor)

    val_pred_tensor = model(X_val_tensor)

    test_pred_tensor = model(X_test_tensor)


# ============================================================
# 17. Tensor 转回 NumPy
# ============================================================

train_pred = (
    train_pred_tensor
    .cpu()
    .numpy()
    .flatten()
)

val_pred = (
    val_pred_tensor
    .cpu()
    .numpy()
    .flatten()
)

test_pred = (
    test_pred_tensor
    .cpu()
    .numpy()
    .flatten()
)


# ============================================================
# 18. 计算 Train / Validation / Test R²
# ============================================================

train_r2 = r2_score(
    y_train.values,
    train_pred,
)

val_r2 = r2_score(
    y_val.values,
    val_pred,
)

test_r2 = r2_score(
    y_test.values,
    test_pred,
)


# ============================================================
# 19. 计算最终测试集 MAE / RMSE
# ============================================================

test_mae = mean_absolute_error(
    y_test.values,
    test_pred,
)

test_rmse = root_mean_squared_error(
    y_test.values,
    test_pred,
)


# ============================================================
# 20. 输出结果
# ============================================================

print("\n==============================")
print("Neural Network 最终结果")
print("==============================")

print(f"Train R² = {train_r2:.3f}")
print(f"Val R²   = {val_r2:.3f}")
print(f"Test R²  = {test_r2:.3f}")

print(f"\nTest MAE  = {test_mae:.3f} MPa")
print(f"Test RMSE = {test_rmse:.3f} MPa")


# ============================================================
# 21. 查看前10个测试预测值
# ============================================================

results = pd.DataFrame({
    "actual_strength": y_test.values,
    "predicted_strength": test_pred,
})

print("\n前10个测试预测结果：")
print(results.head(10))