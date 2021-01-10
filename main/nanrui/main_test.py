from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
from main.nanrui.config import get_param


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


def getData():
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

    test_gen = generator(data,
                         lookback=get_param('lookback'),
                         delay=get_param('delay'),
                         min_index=get_param('train_len') + get_param('val_len'),
                         max_index=None,
                         step=get_param('step'),
                         batch_size=get_param('batch_size'))

    # 准备预测数据
    X = []
    y = []

    howmanybatch = (get_param('test_len') - get_param('lookback')) // get_param('batch_size')  # 需要预测多少个batch
    for test_one in test_gen:
        X.append(test_one[0])
        y.append(test_one[1])
        howmanybatch = howmanybatch - 1
        if howmanybatch == 0:
            break

    return np.vstack(X), np.hstack(y), std[0], mean[0]


if __name__ == '__main__':
    # 获取数据和温度的标准差
    test_X, test_y, std, mean = getData()

    test_y = test_y * std + mean

    # 加载模型
    model = load_model("cnn_rnn.h5")
    test_predict = model.predict(test_X) * std + mean

    # 评估
    print('mae : ' + str(mean_absolute_error(test_y, test_predict)))
    print('rmse : ' + str(sqrt(mean_squared_error(test_y, test_predict))))
    print('r2 : ' + str(r2_score(test_y, test_predict)))

    # 预测结果部分展现
    plt.figure(figsize=(6, 3))
    plt.plot(range(1000), test_y[2000:3000], label='Actual')
    plt.plot(range(1000), test_predict[2000:3000], label='MODEL')
    plt.axis([0, 1000, 25, 45])
    plt.xlabel('Time(h)')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()
