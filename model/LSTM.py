import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from util.datautil import *
from util.modelutil import *

# 检查是否有可用的 GPU 设备。如果有，设置 GPU 内存动态增长，避免一次性占用所有 GPU 内存，并只使用第一块 GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")

seed(1)
tf.random.set_seed(1)

def lstm(model_type, X_train, yuan_X_train):
    #      model type：
    #            1. single-layer LSTM
    #            2. multi-layer LSTM
    #            3. bidirectional LSTM
    global residual_lstm_model
    if model_type == 1:
        # 单层 LSTM
        residual_lstm_model = Sequential()
        residual_lstm_model.add(LSTM(units=50, activation='relu',
                                     input_shape=(X_train.shape[1], 1)))
        residual_lstm_model.add(Dense(units=1))
        original_lstm_model = Sequential()
        original_lstm_model.add(LSTM(units=50, activation='relu',
                    input_shape=(yuan_X_train.shape[1], 5)))
        original_lstm_model.add(Dense(units=5))
    if model_type == 2:
        # 双层 LSTM
        residual_lstm_model = Sequential()
        residual_lstm_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                                     input_shape=(X_train.shape[1], 1)))
        residual_lstm_model.add(LSTM(units=50, activation='relu'))
        residual_lstm_model.add(Dense(1))

        original_lstm_model = Sequential()
        original_lstm_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(yuan_X_train.shape[1], 5)))
        original_lstm_model.add(LSTM(units=50, activation='relu'))
        original_lstm_model.add(Dense(5))
    if model_type == 3:
        # 双向LSTM
        residual_lstm_model = Sequential()
        residual_lstm_model.add(Bidirectional(LSTM(50, activation='relu'),
                                              input_shape=(X_train.shape[1], 1)))
        residual_lstm_model.add(Dense(1))

        original_lstm_model = Sequential()
        original_lstm_model.add(Bidirectional(LSTM(50, activation='relu'),
                                    input_shape=(yuan_X_train.shape[1], 5)))
        original_lstm_model.add(Dense(5))

    return residual_lstm_model, original_lstm_model

# 设置LSTM模型类型及参数
n_timestamp = 10    # 时间步
n_features = 1      # 特征
n_epochs = 50       # 训练轮数
n_batch = 32        # 批次
model_type = 3      # 模型类型

# 读取数据
yuan_data = getData()
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d') 
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

data = getResiduals()
data.index = pd.to_datetime(data['trade_date'])
data = data.drop('trade_date', axis=1)

Lt = pd.read_csv('../temp/ARIMA.csv')
idx = create_data_index(data)

# 数据集划分
training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]
yuan_training_set = yuan_data.iloc[1:idx, :]
yuan_test_set = yuan_data.iloc[idx:, :]

# 数据归一化
sc = MinMaxScaler(feature_range=(0, 1))
yuan_sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(test_set)
yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set)
yuan_testing_set_scaled = yuan_sc.fit_transform(yuan_test_set)

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 数据分割
X_train, y_train = data_split(training_set_scaled, n_timestamp)
yuan_X_train, yuan_y_train = data_split(yuan_training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], yuan_X_train.shape[1], 5)

X_test, y_test = data_split(testing_set_scaled, n_timestamp)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
yuan_X_test, yuan_y_test = data_split(yuan_testing_set_scaled, n_timestamp)
yuna_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], yuan_X_test.shape[1], 5)

# 模型构建
model, yuan_model = lstm(model_type, X_train, yuan_X_train)
# 使用 Adam 优化器和均方误差（MSE）损失函数编译模型
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam,
              loss='mse')
yuan_model.compile(optimizer=adam,
                   loss='mse')

# 模型训练
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),
                    validation_freq=1)
yuan_history = yuan_model.fit(yuan_X_train, yuan_y_train,
                              batch_size=32,
                              epochs=n_epochs,
                              validation_data=(yuan_X_test, yuan_y_test),
                              validation_freq=1)


# 绘制残差模型的训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('residuals: Training and Validation Loss')
plt.legend()
plt.show()

# 绘制原始模型的训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(yuan_history.history['loss'], label='Training Loss')
plt.plot(yuan_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM: Training and Validation Loss')
plt.legend()
plt.show()

# 对测试集数据进行预测 -- 原始数据模型
yuan_predicted_stock_price = yuan_model.predict(yuan_X_test)
yuan_predicted_stock_price = yuan_sc.inverse_transform(yuan_predicted_stock_price)
yuan_predicted_stock_price_list = np.array(yuan_predicted_stock_price[:, 3]).flatten().tolist()
yuan_predicted_stock_price1 = {
    'trade_date': yuan_data.index[idx+10:],
    'close': yuan_predicted_stock_price_list
}
yuan_predicted_stock_price1 = pd.DataFrame(yuan_predicted_stock_price1)
yuan_predicted_stock_price1 = yuan_predicted_stock_price1.set_index(['trade_date'], drop=True)
yuan_real_stock_price = yuan_sc.inverse_transform(yuan_y_test)
yuan_real_stock_price_list = np.array(yuan_real_stock_price[:, 3]).flatten().tolist()
yuan_real_stock_price1 = {
    'trade_date': yuan_data.index[idx+10:],
    'close': yuan_real_stock_price_list
}
yuan_real_stock_price1 = pd.DataFrame(yuan_real_stock_price1)
yuan_real_stock_price1 = yuan_real_stock_price1.set_index(['trade_date'], drop=True)

# 对测试集数据进行预测 -- ARIMA拟合模型
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price_list = np.array(predicted_stock_price[:, 0]).flatten().tolist()

predicted_stock_price1 = {
    'trade_date': data.index[idx+10:],
    'close': predicted_stock_price_list
}
predicted_stock_price1 = pd.DataFrame(predicted_stock_price1)

predicted_stock_price1 = predicted_stock_price1.set_index(['trade_date'], drop=True)

# 对预测结果进行反归一化处理
real_stock_price = sc.inverse_transform(y_test)
# 将 ARIMA 模型的预测结果和残差模型的预测结果相加，得到最终的预测结果
finalpredicted_stock_price = pd.concat([Lt, predicted_stock_price1]).groupby('trade_date')['close'].sum().reset_index()
finalpredicted_stock_price.index = pd.to_datetime(finalpredicted_stock_price['trade_date'])
finalpredicted_stock_price = finalpredicted_stock_price.drop(['trade_date'], axis=1)

# 绘制预测结果图与实际结果图进行模型预测分析
plt.figure(figsize=(10, 6))
plt.plot(yuan_data.iloc[idx+1:, :]['close'], label='Stock Price')
plt.plot(finalpredicted_stock_price['close'], label='Predicted Stock Price')
plt.title('BiLSTM: Stock Price Prediction with ARIMA Residuals')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(yuan_real_stock_price1['close'], label='Stock Price')
plt.plot(yuan_predicted_stock_price1['close'], label='Predicted Stock Price')
plt.title('BiLSTM: Stock Price Prediction on Original Data')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

yhat = yuan_data.iloc[idx+1:, :]['close']

finalpredicted_stock_price, yhat = check_same_length(finalpredicted_stock_price, yhat)

# 模型评估
evaluation_metric(finalpredicted_stock_price['close'], yhat)
