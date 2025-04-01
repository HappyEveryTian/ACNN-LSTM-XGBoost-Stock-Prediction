import matplotlib.pyplot as plt
from util.datautil import *
from util.modelutil import *

data = getData()

residuals = getResiduals()
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals = residuals.drop('trade_date', axis=1)

# 原始数据副本
data_yuan = data
idx = create_data_index(data_yuan)

# 合并ARIMA模型残差的数据
data = pd.merge(data, residuals, on='trade_date')

train_data = data.iloc[1:idx, :]
test_data = data.iloc[idx:, :]

TIME_STEPS = 20

# 数据归一化
train_data, normalize = NormalizeMult(train_data)

# 提取‘close'列并转为二维数组
pollution_data = train_data[:, 3].reshape(len(train_data), 1)

# 划分数据集
train_X, _ = create_dataset(train_data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

# 构建注意力模型
m = cnn_lstm_model(input_dims=7)
model = build_model(m, train_X, train_Y, normalize)

plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 使用混合模型预测并通过XGBoost调优
y_hat, y_test = PredictWithData(test_data, data_yuan, 'close', cnn_lstm_model(input_dims=7), '../temp/stock_model.h5',
                                use_xgb=False)
y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')

time = pd.Series(data.index[idx-1:])
time, y_test = check_same_length(time, y_test)
time, y_hat = check_same_length(time, y_hat)

# 模型评估
evaluation_metric(y_test, y_hat)

# 绘制预测结果对比图
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('cnn-lstm model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()