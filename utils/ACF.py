import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller

csv_path = '../data/nanrui.csv'
df = pd.read_csv(csv_path)
data = df['intemp']
print(adfuller(data))
lag_acf = acf(data, nlags=1000)
plt.style.use('ggplot')
plt.figure(figsize=(6, 3))
plt.plot(lag_acf)
plt.xlim([0,1000])
plt.xlabel('Time Lag (Half Hours)')
plt.ylabel('Autocorrelation')
plt.show()

csv_path = '../data/river.csv'
df = pd.read_csv(csv_path)
data = df['flows']
print(adfuller(data))
lag_acf = acf(data, nlags=1000)
plt.style.use('ggplot')
plt.figure(figsize=(6, 3))
plt.plot(lag_acf)
plt.xlim([0,1000])
plt.xlabel('Time Lag (Days)')
plt.ylabel('Autocorrelation')
plt.show()
