import xgboost as xgb
from keras.layers import Conv1D, Bidirectional, Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.optimizers import Adam
from util.datautil import NormalizeMult, create_dataset, prepare_data

def cnn_lstm_model(input_dims = 13, time_steps = 20, lstm_units = 64):
    inputs = Input(shape=(time_steps, input_dims))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # 一维卷积
    x = Dropout(0.3)(x)  # 防止过拟合

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)  # 双向LSTM
    lstm_out = Dropout(0.3)(lstm_out)

    output = Dense(1, activation='sigmoid')(lstm_out)  # 全连接层
    model = Model(inputs=[inputs], outputs=output)
    return model

def attention_model(input_dims = 13, time_steps = 20, lstm_units = 64):
    inputs = Input(shape=(time_steps, input_dims))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # 一维卷积
    x = Dropout(0.3)(x)     # 防止过拟合

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)    # 双向LSTM
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)        # 注意力加权输出
    attention_mul = Flatten()(attention_mul)        # 扁平化层转化输出为一维数据

    output = Dense(1, activation='sigmoid')(attention_mul)  # 全连接层
    model = Model(inputs=[inputs], outputs=output)
    return model

def PredictWithData(data, data_yuan, name, model, modelname, use_xgb = True):
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testx, _ = create_dataset(data)
    _, testy = create_dataset(data_y)
    if len(testy.shape) == 1:
        testy = testy.reshape(-1, 1)

    model.load_weights(modelname)
    y_hat =  model.predict(testx)
    if use_xgb:
        y_hat, testy = xgb_scheduler(data_yuan)
    return y_hat, testy

def attention_3d_block(inputs, single_attention_vector = False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def xgb_scheduler(data):
    close = data.pop('close')
    data.insert(5, 'close', close)

    train, test = prepare_data(data, n_in=6, n_out=1)
    testy, y_hat2 = walk_forward_validation(train, test)
    return testy, y_hat2

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    for i in range(len(test)):
        testx, testy = test.iloc[i, :-1], test.iloc[i, -1]
        yhat = xgboost_forecast(history, testx)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test.iloc[:, -1], predictions

def xgboost_forecast(train, testx):
    train = np.asarray(train)
    trainx, trainy = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainx, trainy)
    yhat = model.predict(np.asarray([testx]))
    return yhat[0]

def build_model(model, trainx, trainy, normalize):
    # 使用 Adam 优化器，学习率设置为 0.01
    adam = Adam(learning_rate=0.01)
    # 编译模型，使用均方误差（MSE）作为损失函数
    model.compile(optimizer=adam, loss='mse')
    # 训练模型，设置训练轮数为 50，批次大小为 32，使用 10% 的训练数据作为验证集
    m = model.fit([trainx], trainy, epochs=50, batch_size=32, validation_split=0.1)
    # 保存模型文件
    model.save("../temp/stock_model.h5")
    # 保存归一化参数
    np.save("../temp/stock_normalize.npy", normalize)
    return m