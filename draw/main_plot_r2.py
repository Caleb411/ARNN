import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

# 添加数据标签 就是矩形上面的数值
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height+0.0001*height, '%0.2f'%height+'%', ha='center',  va='bottom', fontsize=10, color='black')
        rect.set_edgecolor('white')

if __name__ == "__main__":
    arma = [96.51, 92.35]
    arima = [96.58, 92.37]
    moglstm = [97.88, 91.23]
    combnet = [98.36, 92.50]
    x = [1, 2]
    bar_width = 0.2
    # plt.rcParams['font.family'] = ['Times New Roman']
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    add_labels(plt.bar(x=x,
            height=arma,
            color='#63b2ee',
            edgecolor='white',
            label='ARMA',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width for i in x],
            height=arima,
            color='#76da91',
            edgecolor='white',
            label='ARIMA',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 2 for i in x],
            height=moglstm,
            color='#f8cb7f',
            edgecolor='white',
            label='MogLSTM',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 3 for i in x],
            height=combnet,
            color='#f89588',
            edgecolor='white',
            label='CombNet',
            width=bar_width))
    plt.xticks([i + bar_width + bar_width / 2 for i in x],
               ['Dataset1', 'Dataset2'])
    font = {'size': 15}
    plt.ylabel('R2', fontdict=font)
    plt.ylim(90, 100)
    plt.legend(loc='best')
    plt.show()
