import logging

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from util.datautil import *
from util.modelutil import *

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('../model_metrics.log', encoding='utf-8')
# 创建一个Formatter对象，设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
print(data)

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

m = cnn_lstm_model(input_dims=input_dimens, time_steps=time_steps)
adam = Adam(learning_rate=0.01)
# 编译模型，使用均方误差（MSE）作为损失函数
m.compile(optimizer=adam, loss='mse')

# 训练模型
history = m.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), validation_freq=1)

# x_train_features = pd.DataFrame(x_train_features)
# x_test_features = pd.DataFrame(x_test_features)
# Y_train = pd.Series(Y_train, name='close')
# Y_test = pd.Series(Y_test, name='close')
# new_train_data = pd.merge(x_train_features, Y_train, left_index=True, right_index=True)
# new_test_data = pd.merge(x_test_features, Y_test, left_index=True, right_index=True)
# print(new_train_data)
# print(new_test_data)
#
# _, y_pred = walk_forward_validation(new_train_data, new_test_data)
# print(y_pred)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

y_pred = m.predict(X_test)

# intermediate_layer_model = Model(inputs=m.input,
#                                  outputs=m.get_layer(index=-2).output)
# x_train_features = intermediate_layer_model.predict(X_train)
#
# # 提取测试集特征
# x_test_features = intermediate_layer_model.predict(X_test)
#
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
# eval_set = [(x_train_features, Y_train.ravel()), (x_test_features, Y_test.ravel())]
# model = xgb_model.fit(x_train_features, Y_train.ravel(), eval_set=eval_set, eval_metric="rmse", verbose=False)
#
# # 获取评估结果
# results = model.evals_result()
# epochs = len(results['validation_0']['rmse'])
# x_axis = range(0, epochs)
#
# plt.plot(x_axis, results['validation_0']['rmse'], label='Training Loss')
# plt.plot(x_axis, results['validation_1']['rmse'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
#
# # 使用 XGBoost 模型进行预测
# y_pred = xgb_model.predict(x_test_features)
#
y_pred_reshaped = np.zeros((len(y_pred), testing_set_scaled.shape[1]))
y_pred_reshaped[:, -1] = y_pred.flatten()

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
logger.info(f"模型指标: {metric}")

# 绘制预测结果对比图
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()