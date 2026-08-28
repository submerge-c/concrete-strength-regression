from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. 读取数据
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


# 2. 划分 X 和 y
X = df.drop(columns=["strength"])
y = df["strength"]


# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 4. 标准化输入特征
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)


# 6. 选择 GPU 或 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 7. 把数据移动到 GPU
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# 8. 检查结果
print("使用设备：", device)

print("\n训练输入形状：")
print(X_train_tensor.shape)

print("\n训练目标形状：")
print(y_train_tensor.shape)

print("\n测试输入形状：")
print(X_test_tensor.shape)

print("\n测试目标形状：")
print(y_test_tensor.shape)

print("\nX_train 所在设备：")
print(X_train_tensor.device)

import torch.nn as nn


# 9. 定义神经网络
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


# 10. 创建模型，并放到 GPU
model = ConcreteNet().to(device)

print("\n神经网络结构：")
print(model)


# 11. 随便拿5个样本试运行
sample = X_train_tensor[:5]

with torch.no_grad():
    output = model(sample)

print("\n输入形状：")
print(sample.shape)

print("\n网络输出形状：")
print(output.shape)

print("\n网络当前输出：")
print(output)

# 12. 定义损失函数
criterion = nn.MSELoss()


# 13. 定义优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 14. 训练神经网络
epochs = 2000

for epoch in range(epochs):

    # 进入训练模式
    model.train()

    # 前向传播：模型进行预测
    y_pred = model(X_train_tensor)

    # 计算预测值和真实值之间的误差
    loss = criterion(y_pred, y_train_tensor)

    # 清除上一轮留下的梯度
    optimizer.zero_grad()

    # 反向传播：计算每个参数应该往哪里调整
    loss.backward()

    # 根据梯度修改网络参数
    optimizer.step()

    # 每100轮输出一次
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# 15. 使用训练完成的网络预测测试集
model.eval()

with torch.no_grad():
    y_test_pred_tensor = model(X_test_tensor)


# 16. 从 GPU Tensor 转回 NumPy
y_test_pred = (
    y_test_pred_tensor
    .cpu()
    .numpy()
    .flatten()
)


# 17. 计算模型评价指标
mae = mean_absolute_error(
    y_test.values,
    y_test_pred
)

rmse = root_mean_squared_error(
    y_test.values,
    y_test_pred
)

r2 = r2_score(
    y_test.values,
    y_test_pred
)


print("\nNeural Network 测试结果")
print(f"MAE  = {mae:.3f} MPa")
print(f"RMSE = {rmse:.3f} MPa")
print(f"R²   = {r2:.3f}")