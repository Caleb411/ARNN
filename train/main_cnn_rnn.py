import keras.backend as K
from keras.layers import Multiply, Bidirectional, Conv1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM, GRU
from keras.models import *
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def attention_for_lstm(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_1')(a)
    # 相乘后相加
    output_attention_mul = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([inputs, a_probs]))
    return output_attention_mul


def attention_for_fusion(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(2, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec_2')(a)
    output_attention_mul = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([inputs, a_probs]))
    return output_attention_mul


def maxpooling_for_fusion(inputs):
    output_maxpooliing = Lambda(lambda x: K.max(x, axis=1))(inputs)
    return output_maxpooliing


def model_attention_applied_after_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    # lstm + attention
    lstm_units = 128
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    # (batch_size, sequence_size, hidden_size) -> (batch_size, hidden_size)
    attention_mul = attention_for_lstm(lstm_out)

    # cnn
    conv_out = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs)
    conv_out = Flatten()(conv_out)
    conv_out = Dense(256)(conv_out)

    # attention
    stack_out = Lambda(lambda x: K.stack([x[0], x[1]], axis=1))([attention_mul, conv_out])
    fusion_out = attention_for_fusion(stack_out)
    print(fusion_out.shape)

    output = Dense(1)(fusion_out)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_for_lstm(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1)(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


# 数据生成器函数
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets, [None]


# 绘制学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


SINGLE_ATTENTION_VECTOR = False  # 是否参数共享
APPLY_ATTENTION_BEFORE_LSTM = False  # 是否在lstm前使用Attention
INPUT_DIM = 4
TIME_STEPS = 5

if __name__ == '__main__':

    # 数据读取和预处理
    csv_path = '../data/nanrui.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:], dtype='float')  # 数据删除时间列

    # 参数配置
    train_len = 17434
    val_len = 5811
    test_len = 5811

    # 对数据进行归一化
    mean = data[:train_len].mean(axis=0)
    data -= mean
    std = data[:train_len].std(axis=0)
    data /= std

    # 数据提取
    lookback = 5
    step = 1
    delay = 0
    batch_size = 128
    train_gen = generator(data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=train_len,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(data,
                        lookback=lookback,
                        delay=delay,
                        min_index=train_len,
                        max_index=train_len+val_len,
                        step=step,
                        batch_size=batch_size)

    val_steps = (val_len - lookback) // batch_size

    np.random.seed(666)  # for reproducibility 666

    if APPLY_ATTENTION_BEFORE_LSTM:
        model = model_attention_applied_before_lstm()
    else:
        model = model_attention_applied_after_lstm()

    model.summary()
    model.compile(optimizer=RMSprop(), loss='mae')
    callbacks = [EarlyStopping(patience=5, min_delta=1e-2)]

    history = model.fit(train_gen, steps_per_epoch=100, epochs=10, validation_data=val_gen, validation_steps=val_steps,
                        callbacks=callbacks)
    plot_learning_curves(history)

    # 保存模型
    model.save('./saved/cnn_rnn.h5')
    print('export model saved.')
