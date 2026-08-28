import pandas as pd
from sklearn.model_selection import train_test_split


# 1. 读取数据
data_path = "data/Concrete_Data.xls"
df = pd.read_excel(data_path)


# 2. 把原来很长的列名改短
df.columns = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength"
]


# 3. 查看数据
print("前5行数据：")
print(df.head())

print("\n数据维度：")
print(df.shape)

print("\n缺失值数量：")
print(df.isnull().sum())


# 4. 划分输入特征 X 和目标 y
X = df.drop(columns=["strength"])
y = df["strength"]

print("\nX 的维度：")
print(X.shape)

print("\ny 的维度：")
print(y.shape)


# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n训练集：")
print(X_train.shape)
print(y_train.shape)

print("\n测试集：")
print(X_test.shape)
print(y_test.shape)