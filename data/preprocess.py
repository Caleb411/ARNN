import pandas as pd
import numpy as np

# step 1
with open(r'household_power_consumption.txt') as f1:
    lines = f1.readlines()
    for i in range(0, len(lines)):
        lines[i] = lines[i].replace(';', ',')

with open(r'power.csv', 'a') as f2:
    f2.writelines(lines)

# step 2
o = open(r'power_new.csv', 'a')
with open(r'power.csv') as f1:
    lines = f1.readlines()
    for i in range(0, len(lines)):
        arrs = lines[i].split(',')
        for z, x in enumerate(arrs):
            if x.strip() != '?':
                print(x, end='', file=o)
            if (z != len(arrs) - 1):
                print(',', end='', file=o)
        print(end='', file=o)

o.close()

# step 3
df = pd.read_csv('power_new.csv')
print(df.columns)

df['Global_active_power'].fillna(df['Global_active_power'].mean(), inplace=True)
df['Global_reactive_power'].fillna(df['Global_reactive_power'].mean(), inplace=True)
df['Voltage'].fillna(df['Voltage'].mean(), inplace=True)
df['Global_intensity'].fillna(df['Global_intensity'].mean(), inplace=True)
df['Sub_metering_1'].fillna(df['Sub_metering_1'].mean(), inplace=True)
df['Sub_metering_2'].fillna(df['Sub_metering_2'].mean(), inplace=True)
df['Sub_metering_3'].fillna(df['Sub_metering_3'].mean(), inplace=True)

df.to_csv('power_new_new.csv', index=None)

# step 4
df = np.array(pd.read_csv('power_new_new.csv'))
index = 0
f = open('power.csv', 'a')
for i in range(34587):
    arr = df[index:index + 60, 2:]
    sums = arr.sum(axis=0)
    for j, s in enumerate(sums):
        print(str(round(s, 3)), end='', file=f)
        if (j != 6): print(',', end='', file=f)
    print(file=f)
    index = index + 60
