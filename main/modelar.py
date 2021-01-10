from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K


def attention_for_lstm(inputs):
    # inputs.shape = (batch_size, time_steps, hidden_size)
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec_1')(a)
    # 相乘后相加
    output_attention_mul = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([inputs, a_probs]))
    return output_attention_mul


def get_model(param):
    K.clear_session()
    input = Input(shape=(param['lookback']//param['step'], param['dim']))
    # bilstm + attention
    lstm_units = param['unit']
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(input)
    # (batch_size, sequence_size, hidden_size) -> (batch_size, hidden_size)
    attention_mul = attention_for_lstm(lstm_out)

    # cnn
    conv_out = Conv1D(filters=param['unit'], kernel_size=param['kernel_size'], activation='relu')(input)
    conv_out = Flatten()(conv_out)
    conv_out = Dense(param['unit'])(conv_out)

    # dense
    concat = concatenate([attention_mul, conv_out])
    output = Dense(1)(concat)

    model = Model(input, output)
    return model