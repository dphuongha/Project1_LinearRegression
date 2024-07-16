import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def error(y, y_pred):
    sum_error = 0
    for i in range(0, len(y)):
        sum_error = sum_error + abs(y[i] - y_pred[i])
    return sum_error / len(y)

def error_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, nse, r2

data = pd.read_csv('./milk.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)
k = 5
kf = KFold(n_splits=k, random_state=None)
max = 999999
i = 1
for train_index, test_index in kf.split(dt_Train):
    X_train = dt_Train.iloc[train_index, :7].values
    y_train = dt_Train.iloc[train_index, 7].values
    X_val = dt_Train.iloc[test_index, :7].values
    y_val = dt_Train.iloc[test_index, 7].values

    model = Ridge(max_iter = 800)  # Sử dụng Ridge model
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_error = error(y_train, y_pred_train)
    val_error = error(y_val, y_pred_val)
    total_error = train_error + val_error

    print(f"Train Error: {train_error:.2f}, Validation Error: {val_error:.2f}, Total Error: {train_error + val_error:.2f}")
    if (total_error < max):
        max = total_error
        last = i
        best_model = model  # Chọn mô hình tốt nhất
    i = i + 1

X_test = dt_Test.iloc[:,:7].values
y_test = np.array(dt_Test.iloc[:,7].values)
y_pred = best_model.predict(X_test)
mae, rmse, nse, r2 = error_metrics(y_test, y_pred)

print(f"\nBest Model Total Error: {max:.2f}")
print(f"\nMAE: {mae:.5f},          RMSE: {rmse:.5f},        NSE: {nse:.5f},         R2: {r2:.5f}")
