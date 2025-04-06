import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from util.datautil import evaluation_metric, getOriginData, load_data, create_data_index
from util.logging_config import logger

# load_data('00003.HK')

# 读取数据集
data = getOriginData()
dataSize = data.shape[0] # 获取数据集大小
print(data)

# 重新设置日期为数据集索引
def reset_index_for_dataset(dataset):
    dataset.index = pd.to_datetime(dataset['trade_date'], format='%Y%m%d')

    dataset = dataset.drop(['ts_code', 'trade_date'], axis=1)  # 删除不需要的数据列方便分析

    dataset = pd.DataFrame(dataset, dtype=np.float64)  # 将数据类型设置为64位浮点数

    return dataset

# 划分训练集和测试集
split_radio = 0.95    # 选取前95%数据为训练集，后5%数据为测试集

idx = create_data_index(data)
train_set = data[:idx]
train_set = reset_index_for_dataset(train_set)

test_set = data[idx:]
test_set = reset_index_for_dataset(test_set)

data = reset_index_for_dataset(data)

print(train_set.head())
print(test_set.head())

plt.figure(figsize=(10, 6))
plt.plot(train_set['close'], label='train_set')
plt.plot(test_set['close'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('../save/arima/stock_dataset.png')
plt.show()

stock_data = train_set['close'].copy()

# 检验数据是否平稳
result = adfuller(stock_data)
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print(result)
global temp1

if result[1] > 0.05:
    print("数据非平稳，进行差分处理。")
    # 进行差分
    temp1 = np.diff(train_set['close'], n=1)

    # 绘制差分后的数据
    plt.figure(figsize=(10, 6))
    plt.plot(temp1)
    plt.title("first-order diff")
    plt.xlabel("time")
    plt.ylabel("first-order diff")
    plt.show()

    # ADF检验，检查差分后的数据是否平稳
    result_diff = adfuller(temp1)
    print(f"ADF Statistic after differencing: {result_diff[0]}")
    print(f"p-value after differencing: {result_diff[1]}")

    # 如果差分后数据平稳，继续进行模型拟合
    if result_diff[1] < 0.05:
        print("差分后的数据平稳，可以继续拟合ARIMA模型。")
else:
    print("数据已平稳，无需差分处理。")

# 绘制ACF和PACF图，选择p和q
temp1 = np.diff(train_set['close'], n=1)
plot_acf(temp1)
plot_pacf(temp1)
plt.show()

# 拟合ARIMA模型并提取残差
model1 = ARIMA(endog=train_set['close'], order=(1, 1, 1)).fit()
train_residuals = model1.resid
train_set['rediduals'] = train_residuals
PRED_STEPS = 2  # 每次预测4个时间点
TEST_SIZE = len(test_set)
test_residuals = []
predictions = []
history = train_set['close'].tolist()
for t in range(0, TEST_SIZE, PRED_STEPS):
    model = ARIMA(history, order=(1,1,1)).fit()
    pred = model.forecast(steps=PRED_STEPS)[0]
    predictions.extend(pred)
    actuals = test_set['close'].iloc[t:t+PRED_STEPS].tolist()
    history.extend(actuals)
    residuals = [actuals - pred for actuals, pred in zip(actuals, pred)]
    test_residuals.extend(residuals)
predictions = predictions[:TEST_SIZE]
print(predictions)

predictions1 = {
    'trade_date': test_set.index[:],
    'close': predictions
}
predictions1 = pd.DataFrame(predictions1)
predictions1 = predictions1.set_index(['trade_date'], drop=True)
predictions1.to_csv('../temp/ARIMA.csv')
plt.figure(figsize=(10, 6))
plt.plot(test_set['close'], label='Stock Price')
plt.plot(predictions1, label='Predicted Stock Price')
plt.title('ARIMA: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.savefig('../save/arima/arima-prediction.png')
plt.show()

test_set['rediduals'] = test_residuals
# 提取 test_residuals 中的单个数值
test_residuals_values = [val[0] if isinstance(val, list) else val for val in test_residuals]
train_residuals_df = pd.DataFrame(train_residuals, columns=['residuals'])
test_residuals_df = pd.DataFrame(test_residuals_values, index=test_set.index, columns=['residuals'])
residuals = pd.concat([train_residuals_df, test_residuals_df])
residuals_df = pd.DataFrame(residuals)
fig, ax = plt.subplots(1, 2)
residuals_df.plot(title="Residuals", ax=ax[0])
residuals_df.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
residuals_df.to_csv('../temp/ARIMA_residuals1.csv')

# 计算误差指标
metric = evaluation_metric(test_set['close'], predictions)
logger.info(f"ARIMA模型指标: {metric}")