import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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


# 3. 划分 X 和 y
X = df.drop(columns=["strength"])
y = df["strength"]


# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 5. 创建 Gradient Boosting 模型
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)


# 6. 训练
model.fit(X_train, y_train)


# 7. 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# 8. 模型评价
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

mae = mean_absolute_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)


print("Gradient Boosting 测试结果")
print(f"MAE       = {mae:.3f} MPa")
print(f"RMSE      = {rmse:.3f} MPa")
print(f"训练集 R² = {train_r2:.3f}")
print(f"测试集 R² = {test_r2:.3f}")


# 9. 特征重要性
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\n特征重要性：")
print(feature_importance)