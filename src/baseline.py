import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


# 1. 读取数据
df = pd.read_excel("data/Concrete_Data.xls")


# 2. 简化列名
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


# 3. 输入 X 和目标 y
X = df.drop(columns=["strength"])
y = df["strength"]


# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 5. 创建线性回归模型
model = LinearRegression()


# 6. 训练模型
model.fit(X_train, y_train)


# 7. 预测测试集
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)


# 8. 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print("Linear Regression 测试结果")
print(f"MAE       = {mae:.3f} MPa")
print(f"RMSE      = {rmse:.3f} MPa")
print(f"训练集 R² = {train_r2:.3f}")
print(f"测试集 R² = {test_r2:.3f}")


# 9. 查看部分预测结果
results = pd.DataFrame({
    "actual_strength": y_test.values,
    "predicted_strength": y_pred,
})

print("\n前10个预测结果：")
print(results.head(10))