import matplotlib.pyplot as plt
from keras.optimizers import Adam
from util.utils import *

data = getData()
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

residuals = getResiduals()
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

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
m = attention_model(INPUT_DIMS=7)
# 使用 Adam 优化器，学习率设置为 0.01
adam = Adam(learning_rate=0.01)
# 编译模型，使用均方误差（MSE）作为损失函数
m.compile(optimizer=adam, loss='mse')
# 训练模型，设置训练轮数为 50，批次大小为 32，使用 10% 的训练数据作为验证集
model = m.fit([train_X], train_Y, epochs=50, batch_size=32, validation_split=0.1)
# 保存模型文件
m.save("../temp/stock_model.h5")
# 保存归一化参数
np.save("../temp/stock_normalize.npy", normalize)

plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 使用混合模型预测并通过XGBoost调优
y_hat, y_test = PredictWithData(test_data, data_yuan, 'close', '../temp/stock_model.h5', 7, TIME_STEPS)
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
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()