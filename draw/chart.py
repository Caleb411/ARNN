import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.playStats.descriptive_stats import variance

csv_path = '../data/valve.csv'
df = pd.read_csv(csv_path)

data = np.array(df)
data = np.array(data[:, 2:], dtype='float')

print(variance(data[:, 0]))

plt.figure(figsize=(6, 3))
plt.style.use('seaborn-whitegrid')
plt.plot(range(len(data[:, 0])), data[:, 0], label='ground_truth')
# plt.ylim([-30,30])
plt.xlabel('Time(half hour)')
plt.ylabel('valve')
plt.show()


csv_path = '../data/power.csv'
df = pd.read_csv(csv_path)

data = np.array(df)
data = np.array(data, dtype='float')

print(variance(data[:, 0]))

plt.figure(figsize=(6, 3))
plt.style.use('seaborn-whitegrid')
plt.plot(range(len(data[:, 0])), data[:, 0], label='ground_truth')
# plt.ylim([-30,30])
plt.xlabel('Time(hour)')
plt.ylabel('power')
plt.show()
