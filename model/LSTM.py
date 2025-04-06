import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from util.datautil import *
from util.logging_config import logger
from util.modelutil import *

data = getData()

residuals = getResiduals()
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

# 原始数据副本
data_yuan = data
idx = create_data_index(data_yuan)

# 合并ARIMA模型残差的数据
data = pd.merge(data, residuals, on='trade_date')
# 将'close'列调整到最后一列
index = data.columns.shape[0] - 1
close = data.pop('close')
data.insert(index, 'close', close)

train_data = data[1:idx]
test_data = data[idx:]

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

time_steps = 20
input_dimens = data.columns.shape[0] - 1

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_data)
testing_set_scaled = sc.fit_transform(test_data)

X_train, Y_train = data_split(training_set_scaled, time_steps)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], input_dimens)
X_test, Y_test = data_split(testing_set_scaled, time_steps)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], input_dimens)

# 模型构建
model = lstm_model(input_dims=input_dimens, time_steps=time_steps)
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mse')

# 训练模型
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), validation_freq=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('../save/lstm/lstm-loss.png')
plt.show()

predictions = model.predict(X_test)

y_pred_reshaped = np.zeros((len(predictions), testing_set_scaled.shape[1]))
y_pred_reshaped[:, -1] = predictions.flatten()

y_test_reshaped = np.zeros((len(Y_test), testing_set_scaled.shape[1]))
y_test_reshaped[:, -1] = Y_test

# 进行反归一化
y_hat = sc.inverse_transform(y_pred_reshaped)[:, -1]
y_test = sc.inverse_transform(y_test_reshaped)[:, -1]

time = pd.Series(data.index[idx-1:])
time, y_test = check_same_length(time, y_test)
time, y_hat = check_same_length(time, y_hat)

# 模型评估
metric = evaluation_metric(y_test, y_hat)
logger.info(f"LSTM模型指标: {metric}")

# 绘制预测结果对比图
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('LSTM model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('../save/lstm/lstm-prediction.png')
plt.show()