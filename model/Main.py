import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
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

m = attention_model(input_dims=input_dimens, time_steps=time_steps)
adam = Adam(learning_rate=0.01)
# 编译模型，使用均方误差（MSE）作为损失函数
m.compile(optimizer=adam, loss='mse')

# 训练模型
history = m.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), validation_freq=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('../save/hybrid/hybrid-loss.png')
plt.show()

intermediate_layer_model = Model(inputs=m.input,
                                 outputs=m.get_layer(index=-2).output)
x_train_features = intermediate_layer_model.predict(X_train)

# 提取测试集特征
x_test_features = intermediate_layer_model.predict(X_test)

# 进行滚动预测
predictions = []
for i in range(len(x_test_features)):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    xgb_model.fit(x_train_features, Y_train.ravel())
    pred = xgb_model.predict(x_test_features[i].reshape(1, -1))
    print(i + 1, '>expected=%.6f, predicted=%.6f' % (Y_test[i], pred[0]))
    predictions.append(pred[0])
    x_train_features = np.vstack([x_train_features, x_test_features[i]])
    Y_train = np.append(Y_train, Y_test[i])

y_pred_reshaped = np.zeros((len(predictions), testing_set_scaled.shape[1]))
y_pred_reshaped[:, -1] = predictions

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
logger.info(f"混合模型指标: {metric}")

# 绘制预测结果对比图
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('../save/hybrid/hybrid-prediction.png')
plt.show()