from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump,load
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
from math import sqrt

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


def getData():
    # 数据读取和预处理
    csv_path = '../data/nanrui.csv'
    df = pd.read_csv(csv_path)
    data = np.array(df)
    data = np.array(data[:, 2:], dtype='float')  # 数据删除时间列

    # 对数据进行归一化
    mean = data[:17434].mean(axis=0)
    data -= mean
    std = data[:17434].std(axis=0)
    data /= std

    # 数据提取
    lookback = 5
    step = 1
    delay = 0
    batch_size = 128

    test_gen = generator(data,
                         lookback=lookback,
                         delay=delay,
                         min_index=23245,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # 准备预测数据
    X = []
    y = []

    howmanybatch = (5811 - lookback) // batch_size  # 需要预测多少个batch
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
    cnn_rnn_model = load_model("../saved/cnn_rnn.h5")
    svm_model = load("../saved/svm.pkl")

    # 预测
    svm_test_predict = svm_model.predict(test_X[:, :, 0])
    svm_test_predict = svm_test_predict.reshape(-1, 1)
    cnn_rnn_test_predict = cnn_rnn_model.predict(test_X)

    X = []
    Y = []

    for i in range(11):
        a = i / 10
        X.append(a)
        b = (10 - i) / 10
        print(a, b)
        test_predict = (svm_test_predict*a+cnn_rnn_test_predict*b) * std + mean

        r2_result = r2_score(test_y, test_predict)
        Y.append(r2_result)

        # 误差评估
        print('mae : ' + str(mean_absolute_error(test_y, test_predict)))
        print('rmse : ' + str(sqrt(mean_squared_error(test_y, test_predict))))
        print('r2 : ' + str(r2_result))

    plt.style.use('seaborn-whitegrid')
    plt.plot(X,Y,'-*')
    font = {'size': 15}
    plt.ylabel('R2', fontdict=font)
    plt.xlabel('α')
    plt.show()

    # # 预测结果部分展现
    # plt.figure(figsize=(6, 3))
    # plt.plot(range(1000), test_y[2000:3000], label='Actual')
    # plt.plot(range(1000), test_predict[2000:3000], label='SVM_CNN_RNN')
    # plt.axis([0, 1000, 0, 40])
    # plt.xlabel('Time(h)')
    # plt.ylabel('Temperature')
    # plt.legend()
    # plt.show()
