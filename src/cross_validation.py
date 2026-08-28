import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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


# 3. 仍然保留20%测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# 4. 定义三个模型
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


# 5. 5折交叉验证
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# 6. 评价指标
scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


# 7. 分别验证三个模型
for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    mean_r2 = scores["test_r2"].mean()
    std_r2 = scores["test_r2"].std()

    mean_mae = -scores["test_mae"].mean()
    mean_rmse = -scores["test_rmse"].mean()

    print(f"\n{name}")
    print(f"平均 R²   = {mean_r2:.3f}")
    print(f"R² 标准差 = {std_r2:.3f}")
    print(f"平均 MAE  = {mean_mae:.3f} MPa")
    print(f"平均 RMSE = {mean_rmse:.3f} MPa")