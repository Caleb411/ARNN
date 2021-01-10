import numpy as np
import pandas as pd
from main.nanrui.config import get_param, get_params
import tensorflow as tf
import os
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping


# 设定随机种子
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# 数据生成器函数
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay
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
            targets[j] = data[rows[j] + delay - 1][0]
        yield samples, targets, [None]


def get_model(param):
    K.clear_session()
    input = Input(shape=(param['lookback'], param['dim']))
    lstm_out = LSTM(param['unit'], return_sequences=True)(input)
    x = Lambda(lambda a: K.reshape(a, (-1, get_param('unit') * get_param('lookback'))))(lstm_out)
    output = Dense(1)(x)
    model = Model(input, output)
    return model


if __name__ == '__main__':
    # 数据读取和预处理
    csv_path = '../../data/nanrui.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:], dtype='float')  # 数据删除时间列

    # 对数据进行归一化
    mean = data[:get_param('train_len')].mean(axis=0)
    data -= mean
    std = data[:get_param('train_len')].std(axis=0)
    data /= std

    # 数据提取
    train_gen = generator(data,
                          lookback=get_param('lookback'),
                          delay=get_param('delay'),
                          min_index=0,
                          max_index=get_param('train_len'),
                          shuffle=True,
                          step=get_param('step'),
                          batch_size=get_param('batch_size'))
    val_gen = generator(data,
                        lookback=get_param('lookback'),
                        delay=get_param('delay'),
                        min_index=get_param('train_len'),
                        max_index=get_param('train_len') + get_param('val_len'),
                        step=get_param('step'),
                        batch_size=get_param('batch_size'))

    test_gen = generator(data,
                         lookback=get_param('lookback'),
                         delay=get_param('delay'),
                         min_index=get_param('train_len') + get_param('val_len'),
                         max_index=None,
                         step=get_param('step'),
                         batch_size=get_param('batch_size'))

    val_steps = (get_param('val_len') - get_param('lookback')) // get_param('batch_size')

    # 准备预测数据
    X = []
    y = []

    howmanybatch = (get_param('test_len') - get_param('lookback')) // get_param('batch_size')
    for test_one in test_gen:
        X.append(test_one[0])
        y.append(test_one[1])
        howmanybatch = howmanybatch - 1
        if howmanybatch == 0:
            break

    test_y = np.hstack(y) * std[0] + mean[0]

    set_seed(666)

    model = get_model(get_params())
    model.summary()
    model.compile(optimizer='adam', loss='mae')
    callbacks = [EarlyStopping(patience=5, min_delta=1e-2)]

    history = model.fit(train_gen,
                        steps_per_epoch=get_param('steps_per_epoch'),
                        epochs=get_param('epochs')//2,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        callbacks=callbacks)

    test_predict = model.predict(np.vstack(X)) * std[0] + mean[0]

    # 评估
    print("mae\t%f" % mean_absolute_error(test_y, test_predict))
    print("rmse\t%f" % sqrt(mean_squared_error(test_y, test_predict)))
    print("r2\t%f" % r2_score(test_y, test_predict))
