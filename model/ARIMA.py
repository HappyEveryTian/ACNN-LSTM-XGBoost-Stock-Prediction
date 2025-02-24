import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from util.utils import evaluation_metric
"""
数据集文件来自沪深股票的每日指标类型
"""

# 读取数据集
data = pd.read_csv('../dataset/601988.SH.csv')
dataSize = data.shape[0] # 获取数据集大小

# 重新设置日期为数据集索引
def reset_index_for_dataset(dataset):
    dataset.index = pd.to_datetime(dataset['trade_date'], format='%Y%m%d')

    dataset = dataset.drop(['ts_code', 'trade_date'], axis=1)  # 删除不需要的数据列方便分析

    dataset = pd.DataFrame(dataset, dtype=np.float64)  # 将数据类型设置为64位浮点数

    return dataset

# 划分训练集和测试集
split_radio = 0.95    # 选取前split_radio数据为训练集，后(1 - split_radio)数据为测试集

train_set = data.loc[:int(dataSize*split_radio)]
train_set = reset_index_for_dataset(train_set)

test_set = data.loc[int(dataSize*split_radio):]
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
plt.show()

stock_data = train_set['close']

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

history = [x for x in train_set['close']]
predictions = list()
for t in range(test_set.shape[0]):
    # 根据p,d,q参数设置模型并进行训练
    model1 = ARIMA(history, order=(1, 1, 0))
    model_fit = model1.fit()
    yhat = model_fit.forecast(steps=1)  # 预测一步
    yhat = np.float64(yhat[0])  # 提取预测值
    predictions.append(yhat)
    # 将实际值添加到history，以便下次迭代使用
    history.append(test_set['close'][t])

# 绘制预测结果图与实际结果图进行模型预测分析
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
plt.show()

# 计算误差指标
evaluation_metric(test_set['close'], predictions)

# 拟合ARIMA模型并提取残差
model1 = ARIMA(endog=data['close'], order=(1, 1, 0)).fit()
residuals = pd.DataFrame(model1.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
residuals.to_csv('../temp/ARIMA_residuals1.csv')