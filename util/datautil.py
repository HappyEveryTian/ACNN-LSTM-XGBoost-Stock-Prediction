import pandas as pd
from keras.layers.core import *
from sklearn import metrics
import tushare as ts
import configparser

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

def create_dataset(dataset, look_back=20):
    datax, datay = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), :]
        datax.append(a)
        datay.append(dataset[i + look_back, :])
    trainx = np.array(datax)
    train_y = np.array(datay)

    return trainx, train_y

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
    data = pd.read_csv('../dataset/601988.SH.csv')
    return data

def getData():
    data = getOriginData()
    data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
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