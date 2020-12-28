import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from joblib import dump


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=True):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


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
    mean = data[:17434].mean(axis=0)
    data -= mean
    std = data[:17434].std(axis=0)
    data /= std

    X_train, y_train = multivariate_data(data[:, 0], data[:, 0], 0, train_len, 5, 0, 1, True)
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)

    # X_valid, y_valid = multivariate_data(data[:, 0], data[:, 0], train_len, train_len+val_len, 5, 0, 1, True)
    # svr.score(X_valid, y_valid)
    #
    # y_predict = svr.predict(X_valid)
    # plt.plot(range(len(y_predict)), y_predict, label='predict')
    # plt.plot(range(len(y_valid)), y_valid, label='valid')
    # plt.legend()
    # plt.show()

    dump(svr, '../saved/svm.pkl')
