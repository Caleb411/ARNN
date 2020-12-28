import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'

# 添加数据标签 就是矩形上面的数值
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height+0.0001*height, '%0.2f'%height+'%', ha='center',  va='bottom', fontsize=12, color='black')
        rect.set_edgecolor('white')

if __name__ == "__main__":
    y1 = [92.50]
    y2 = [89.87]
    y3 = [91.03]
    y4 = [87.50]
    y5 = [92.20]
    y6 = [92.28]
    y7 = [92.50]
    x = [1]
    bar_width = 0.1
    # plt.rcParams['font.family'] = ['Times New Roman']
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    add_labels(plt.bar(x=x,
            height=y1,
            color='#63b2ee',
            edgecolor='white',
            label='Combw/oSVM',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width for i in x],
            height=y2,
            color='#76da91',
            edgecolor='white',
            label='Combw/oSVM_CNN',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 2 for i in x],
            height=y3,
            color='#f8cb7f',
            edgecolor='white',
            label='Combw/oCNN',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 3 for i in x],
            height=y4,
            color='#f89588',
            edgecolor='white',
            label='SVM',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 4 for i in x],
            height=y5,
            color='#7cd6cf',
            edgecolor='white',
            label='CombNet-GRU',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 5 for i in x],
            height=y6,
            color='#9192ab',
            edgecolor='white',
            label='CombNet-max',
            width=bar_width))
    add_labels(plt.bar(x=[i + bar_width * 6 for i in x],
            height=y7,
            color='#7898e1',
            edgecolor='white',
            label='CombNet',
            width=bar_width))
    plt.xticks([],[])
    font = {'size': 15}
    plt.ylabel('R2',fontdict=font)
    plt.ylim(86, 100)
    plt.legend(loc='upper right')
    plt.show()
