import matplotlib.pyplot as plt
from util.datautil import *
from util.modelutil import walk_forward_validation

data = getData()

# 将'close'列调整到最后一列
close = data.pop('close')
data.insert(5, 'close', close)

idx = create_data_index(data)

data1 = data.iloc[idx+1:, 5]
residuals = getResiduals()
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')
merge_data = pd.merge(data, residuals, on='trade_date')
time = pd.Series(data.index[idx+1:])

Lt = pd.read_csv('../temp/ARIMA.csv')
Lt = Lt.drop('trade_date', axis=1)
Lt = np.array(Lt)
Lt = Lt.flatten().tolist()

train, test = prepare_data(merge_data, n_in=6, n_out=1)

y, yhat = walk_forward_validation(train, test)

time, y = check_same_length(time, y)
time, yhat = check_same_length(time, yhat)

plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Residuals')
plt.plot(time, yhat, label='Predicted Residuals')
plt.title('ARIMA+XGBoost: Residuals Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

finalpredicted_stock_price = [i + j for i, j in zip(Lt, yhat)]

evaluation_metric(data1, finalpredicted_stock_price)

plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('ARIMA+XGBoost: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
