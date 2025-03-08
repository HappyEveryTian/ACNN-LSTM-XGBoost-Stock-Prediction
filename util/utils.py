import pandas as pd
import xgboost as xgb
from keras.layers import Conv1D, Bidirectional, Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from sklearn import metrics


def evaluation_metric(y_test,y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test,y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)

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
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

def attention_model(INPUT_DIMS = 13, TIME_STEPS = 20, lstm_units = 64):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # 一维卷积
    x = Dropout(0.3)(x)     # 防止过拟合

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)    # 双向LSTM
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)        # 注意力加权输出
    attention_mul = Flatten()(attention_mul)        # 扁平化层转化输出为一维数据

    output = Dense(1, activation='sigmoid')(attention_mul)  # 全连接层
    model = Model(inputs=[inputs], outputs=output)
    return model

def PredictWithData(data, data_yuan, name, modelname, INPUT_DIMS = 13, TIME_STEPS = 20):
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    model = attention_model(INPUT_DIMS)
    model.load_weights(modelname)
    y_hat =  model.predict(testX)
    testY, y_hat = xgb_scheduler(data_yuan, y_hat)
    return y_hat, testY

def attention_3d_block(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def create_data_index(data):
    dataSize = data.shape[0]
    split_radio = 0.95
    idx = int(dataSize * split_radio)
    return idx

def xgb_scheduler(data, y_hat):
    close = data.pop('close')
    data.insert(5, 'close', close)

    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def prepare_data(series, n_test, n_in, n_out):
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

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test.iloc[:, -1], predictions

def xgboost_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def check_same_length(y1, y2):
    if len(y1) != len(y2):
        min_length = min(len(y1), len(y2))
        y1 = y1[:min_length]
        y2 = y2[:min_length]
    return y1, y2

def getData():
    data = pd.read_csv('../dataset/01810.HK.csv')
    return data

def getResiduals():
    data = pd.read_csv('../temp/ARIMA_residuals1.csv')
    return data