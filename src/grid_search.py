import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


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


# 2. X 和 y
X = df.drop(columns=["strength"])
y = df["strength"]


# 3. 保留最终测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 4. 创建基础模型
model = GradientBoostingRegressor(
    random_state=42
)


# 5. 定义要尝试的参数
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
}


# 6. 定义5折交叉验证
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# 7. 网格搜索
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)


# 8. 搜索最优参数
grid_search.fit(X_train, y_train)


print("\n最佳参数：")
print(grid_search.best_params_)

print("\n交叉验证最佳 RMSE：")
print(f"{-grid_search.best_score_:.3f} MPa")


# 9. 取得最佳模型
best_model = grid_search.best_estimator_


# 10. 在最终测试集上预测
y_pred = best_model.predict(X_test)


# 11. 最终评价
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n最终测试集结果")
print(f"MAE  = {mae:.3f} MPa")
print(f"RMSE = {rmse:.3f} MPa")
print(f"R²   = {r2:.3f}")