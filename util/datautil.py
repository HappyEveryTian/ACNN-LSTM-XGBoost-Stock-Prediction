import pandas as pd
from keras.layers.core import *
from sklearn import metrics
import tushare as ts
import configparser

def ema_process(df, column_name):
    df['EMA_5'] = df[column_name].ewm(span=5, adjust=False, min_periods=1).mean()
    return df

def sma_process(df, column_name):
    # 对前五行使用扩展窗口计算 SMA
    first_five_sma = df[column_name].expanding(min_periods=1).mean().head(5)

    # 对剩余的行使用窗口大小为 5 计算 SMA
    remaining_sma = df[column_name].rolling(window=5).mean().tail(len(df) - 5)

    # 合并两个结果
    sma_series = pd.concat([first_five_sma, remaining_sma])

    # 将 SMA 指标作为新的列加入到数据中
    df[f'{column_name}_SMA'] = sma_series
    return df

def evaluation_metric(y_test, y_hat):
    mse = metrics.mean_squared_error(y_test, y_hat)
    rmse = mse**0.5
    mae = metrics.mean_absolute_error(y_test, y_hat)
    r2 = metrics.r2_score(y_test, y_hat)
    print('MSE: %.5f' % mse)
    print('RMSE: %.5f' % rmse)
    print('MAE: %.5f' % mae)
    print('R2: %.5f' % r2)
    return { 'mse': '%.5f' % mse, 'rmse': '%.5f' % rmse, 'mae': '%.5f' % mae, 'r2': '%.5f' % r2 }

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

def create_data_index(data):
    data_size = data.shape[0]
    split_radio = 0.95
    idx = int(data_size * split_radio)
    return idx

def check_same_length(y1, y2):
    if len(y1) != len(y2):
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]
    return y1, y2

def getOriginData():
    data = pd.read_csv('../dataset/00003.HK.csv')
    return data

def getData():
    data = getOriginData()
    data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data.loc[:, ['open', 'high', 'low', 'pre_close', 'close', 'vol', 'amount']]
    data = data.fillna(method="ffill")
    return data

def getResiduals():
    data = pd.read_csv('../temp/ARIMA_residuals1.csv')
    return data

def load_data(ts_code):
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini')
    token = config.get('tushare', 'token')

    # 获取数据集
    pro = ts.pro_api(token)

    # 拉取数据
    df = pro.hk_daily(ts_code=ts_code)

    df = pd.DataFrame(df)
    df['ts_code'] = ts_code
    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)
    # 保存数据集到本地
    df.to_csv('../dataset/' + ts_code + '.csv')