from tensorflow.keras.callbacks import EarlyStopping
from main.nanrui.config import get_params, get_param
from main.modelcnn import get_model
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random


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


# 绘制学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


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

    val_steps = (get_param('val_len') - get_param('lookback')) // get_param('batch_size')

    set_seed(666)  # for reproducibility 666

    model = get_model(get_params())

    model.summary()
    model.compile(optimizer='adam', loss='mae')
    callbacks = [EarlyStopping(patience=5, min_delta=1e-2)]

    history = model.fit(train_gen,
                        steps_per_epoch=get_param('steps_per_epoch'),
                        epochs=9,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        callbacks=callbacks)
    plot_learning_curves(history)

    # 保存模型
    model.save('cnn_rnn.h5')
    print('export model saved.')
