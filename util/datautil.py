import pandas as pd
from keras.layers.core import *
from sklearn import metrics
import tushare as ts

def evaluation_metric(y_test, y_hat):
    mse = metrics.mean_squared_error(y_test, y_hat)
    rmse = mse**0.5
    mae = metrics.mean_absolute_error(y_test, y_hat)
    r2 = metrics.r2_score(y_test, y_hat)
    print('MSE: %.5f' % mse)
    print('RMSE: %.5f' % rmse)
    print('MAE: %.5f' % mae)
    print('R2: %.5f' % r2)

def NormalizeMult(data):
    data = np.array(data)
    normalize = np.arange(2 * data.shape[1], dtype='float64')
    normalize = normalize.reshape(data.shape[1], 2)
    for i in range(0, data.shape[1]):
        list = data[:, i]
        listlow, listhigh = np.percentile(list, [0, 100])
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta
    return data, normalize

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

def prepare_data(series, n_in, n_out):
    idx = create_data_index(series)
    values = series.values
    supervised_data = series_to_supervised(values, n_in, n_out)
    train, test = supervised_data.loc[:idx, :], supervised_data.loc[idx:, :]
    return train, test

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def check_same_length(y1, y2):
    if len(y1) != len(y2):
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]
    return y1, y2

def getData():
    data = pd.read_csv('../dataset/601988.SH.csv')
    return data

def getResiduals():
    data = pd.read_csv('../temp/ARIMA_residuals1.csv')
    return data

def load_data(ts_code):
    # 获取数据集
    pro = ts.pro_api('bacd53cd2890aac36761bcf9a29ea6b60d061bce134c8c880b4a9dc9')

    # 拉取数据
    df = pro.hk_daily(ts_code='00003.HK', start_date="19950302")

    df = pd.DataFrame(df)
    df['ts_code'] = ts_code
    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)
    df.to_csv('../dataset/' + ts_code + '.csv')