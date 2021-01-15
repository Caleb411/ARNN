from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

# 数据读取和预处理
csv_path = '../../data/valve.csv'
df = pd.read_csv(csv_path)
data = np.array(df)
data = np.array(data[:, 2:3], dtype='float')  # 数据删除时间列

# 对数据进行归一化
train_len = 21792
mean = data[:train_len].mean(axis=0)
data -= mean
std = data[:train_len].std(axis=0)
data /= std

train_data = data[:train_len]   # (21792,1)
test_data = data[train_len:]*std[0]+mean[0]    # (7264,1)

# 加载模型
model = load_model("cnn_rnn.h5")

# 多步预测
len = 100
history = train_data.reshape(-1).tolist() # (21792,)
predict_data = list()
for t in tqdm(range(len)):
    test_predict_old = model.predict(np.reshape(np.array(history[-48:]), (1,48,1)))
    test_predict_new = test_predict_old * std[0] + mean[0]
    history.append(float(test_predict_old))
    predict_data.append(float(test_predict_new))

# 预测结果部分展现
plt.figure(figsize=(6, 3))
plt.plot(range(len), test_data.reshape(-1).tolist()[:len], label='Actual')
plt.plot(range(len), predict_data[:len], label='MODEL')
plt.axis([0, len, 25, 45])
plt.xlabel('Time(h)')
plt.ylabel('Temperature')
plt.legend()
plt.show()
