from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from main.valve.config import get_param
from sklearn.linear_model import Ridge

# 生成测试集
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=True):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices].reshape(-1))

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


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


if __name__ == '__main__':
    # 数据读取和预处理
    csv_path = '../../data/valve.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:], dtype='float')  # 数据删除时间列

    # 对数据进行归一化
    mean = data[:get_param('train_len')].mean(axis=0)
    data -= mean
    std = data[:get_param('train_len')].std(axis=0)
    data /= std

    X_train, y_train = multivariate_data(data, data[:, 0], 0,
                                         get_param('train_len'),
                                         get_param('lookback'),
                                         get_param('delay') - 1,
                                         get_param('step'))

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

    lin_reg = Ridge()
    lin_reg.fit(X_train, y_train)

    test_predict = lin_reg.predict(np.vstack(X).reshape(-1, get_param('lookback')*get_param('dim'))) * std[0] + mean[0]

    # 误差评估
    print('mae : ' + str(mean_absolute_error(test_y, test_predict)))
    print('rmse : ' + str(sqrt(mean_squared_error(test_y, test_predict))))
    print('r2 : ' + str(r2_score(test_y, test_predict)))
