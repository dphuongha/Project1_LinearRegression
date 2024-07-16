import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #do hieu suat mo hinh
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('./milk.csv')

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

X_train = dt_Train.iloc[:,:7]
y_train= dt_Train.iloc[:,7]

X_test = dt_Test.iloc[:,:7]
y_test=dt_Test.iloc[:,7]

reg = Ridge(alpha=0.5, max_iter = 800) 
reg.fit(X_train, y_train) #Huấn luyện nó trên dữ liệu huấn luyện, tính trọng số w và b
#y = wx + b

#Dự đoán trên tập Test (Kiểm thử)
y_pred = reg.predict(X_test)

#print('Trọng số w = ', reg.coef_)
#print('\nb = ', reg.intercept_)


print("Ridge: \n")
# Độ đo Nash-Sutcliffe Efficiency (NSE)
nse = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
print("NSE: %.5f" % nse)

# Độ đo R2
r2 = r2_score(y_test, y_pred)
print("R2: %.5f" % r2)

# Độ đo Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: %.5f" % mae)
# Độ đo Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %.5f" % rmse)
print('\n')

#So sánh giữa giá trị thực (y_test) với giá trị dự đoán (y_pred) và sự chênh lệch tuyệt đối giữa chúng
y = np.array(y_test)
print("Thuc te              Du doan             Chenh lech")
for i in range(0, len(y)):
   print("%.2f" % y[i],"       ", y_pred[i], "     ", abs(y[i] -y_pred[i]))







