params = {
    'train_len': 17434,     # 训练集长度
    'val_len': 5811,        # 验证集长度
    'test_len': 5811,       # 测试集长度

    'lookback': 48,         # 采样长度
    'dim': 4,               # 每个时间点数据的维度
    'step': 1,              # 采样间隔
    'delay': 3,             # 延迟步数 horizon>=1 [3,6,12,24]
    'batch_size': 128,      # 批量大小

    'unit': 256,            # 神经网络超参数
    'kernel_size': 3,       # 卷积核的大小
    'hw': 3,                # AR超参数

    'steps_per_epoch': 100, # 每轮训练的批量数
    'epochs': 10            # 训练的轮数
}


def get_param(param):
    return params[param]

def get_params():
    return params
