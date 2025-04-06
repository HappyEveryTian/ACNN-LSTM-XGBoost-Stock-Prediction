from keras.layers import Conv1D, Bidirectional, Multiply, MaxPooling1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

def lstm_model(input_dims, time_steps = 10, model_type = 3, units = 64):
    global lstm_out
    inputs = Input(shape=(time_steps, input_dims))
    if model_type == 1:
        lstm_out = LSTM(units=units, activation='relu', return_sequences=True)(inputs)
    if model_type == 2:
        lstm_out = LSTM(units=units, activation='relu', return_sequences=True)(inputs)
        lstm_out = LSTM(units=units, activation='relu', return_sequences=True)(lstm_out)
    if model_type == 3:
        lstm_out = Bidirectional(LSTM(units=units, activation='relu', return_sequences=True))(inputs)
    lstm_out = Dense(units=input_dims)(lstm_out)
    flatten_out = Flatten()(lstm_out)
    model_out = Dense(1, activation='sigmoid')(flatten_out)
    model = Model(inputs=[inputs], outputs=model_out)
    return model

def cnn_lstm_model(input_dims = 13, time_steps = 20, lstm_units = 64):
    inputs = Input(shape=(time_steps, input_dims))

    cnn_out = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    pool_out = MaxPooling1D(pool_size=2)(cnn_out)
    bilstm_out = Bidirectional(LSTM(units=lstm_units, activation='relu', input_shape=(time_steps, input_dims), return_sequences=True))(pool_out)
    bilstm_out = Dense(units=input_dims)(bilstm_out)
    flatten_out = Flatten()(bilstm_out)
    outputs = Dense(1, activation='sigmoid')(flatten_out)

    model = Model(inputs=[inputs], outputs=outputs)
    return model

def attention_model(input_dims = 6, time_steps = 20, lstm_units = 64):
    inputs = Input(shape=(time_steps, input_dims))

    cnn_out = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    pool_out = MaxPooling1D(pool_size=2)(cnn_out)
    bilstm_out = Bidirectional(LSTM(units=lstm_units, activation='relu', input_shape=(time_steps, input_dims), return_sequences=True))(pool_out)
    bilstm_out = Dense(units=input_dims)(bilstm_out)
    attention_mul = attention_3d_block(bilstm_out)
    flatten_out = Flatten()(attention_mul)
    outputs = Dense(1, activation='sigmoid')(flatten_out)

    model = Model(inputs=[inputs], outputs=outputs)
    return model

def attention_3d_block(inputs):
    features = K.int_shape(inputs)[-1]

    # 对每个特征独立生成时间步权重
    a = Conv1D(filters=1, kernel_size=1, activation='tanh', use_bias=False)(inputs)
    a = Flatten()(a)
    a = Activation('softmax')(a)  # 时间步权重
    a = RepeatVector(features)(a)
    a = Permute((2, 1))(a)

    return Multiply()([inputs, a])