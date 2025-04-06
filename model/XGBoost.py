import matplotlib.pyplot as plt
from util.logging_config import logger
from util.datautil import *
from util.modelutil import *
import xgboost as xgb

data = getData()

residuals = getResiduals()
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

# 原始数据副本
data_yuan = data
idx = create_data_index(data_yuan)

# 合并ARIMA模型残差的数据
data = pd.merge(data, residuals, on='trade_date')
# 生成滞后特征
def add_lags(data, target_col, lags):
    for lag in lags:
        data[f'{target_col}_lag{lag}'] = data[target_col].shift(lag)
    return data.dropna()

# data = add_lags(data, 'close', lags=[1,7])

# 将'close'列调整到最后一列
index = data.columns.shape[0] - 1
close = data.pop('close')
data.insert(index, 'close', close)

train_data = np.asarray(data[:idx])
test_data = np.asarray(data[idx:])

trainx, trainy = train_data[:, :-1], train_data[:, -1]
testx, testy = test_data[:, :-1], test_data[:, -1]

# 滚动预测
# predictions = []
# for i in range(len(testx)):
#     xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
#     xgb_model.fit(trainx, trainy)
#     pred = xgb_model.predict(testx[i].reshape(1, -1))
#     print(i + 1, '>expected=%.6f, predicted=%.6f' % (testy[i], pred))
#     predictions.append(pred[0])
#     trainx = np.vstack([trainx, testx[i]])
#     trainy = np.append(trainy, testy[i])
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=16)
xgb_model.fit(trainx, trainy)
predictions = xgb_model.predict(testx)

time = pd.Series(data.index[idx-1:])
time, testy = check_same_length(time, testy)
time, y_pred = check_same_length(time, predictions)

# 模型评估
metric = evaluation_metric(testy, y_pred)
logger.info(f"XGBoost模型指标: {metric}")

# 绘制预测结果对比图
plt.plot(time, testy, label='True')
plt.plot(time, y_pred, label='Prediction')
plt.title('XGBoost model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('../save/xgboost/xgboost-prediction.png')
plt.show()