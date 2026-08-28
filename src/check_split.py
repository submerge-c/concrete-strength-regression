import pandas as pd

from sklearn.model_selection import train_test_split


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


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2,
    random_state=42,
)


print("Train:")
print(y_train.describe())

print("\nValidation:")
print(y_val.describe())

print("\nTest:")
print(y_test.describe())