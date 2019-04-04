from keras import backend as K
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import LSTM, Dropout, Add

from layers.layers import LayerNormalization


class DecoderLayer:
    def __init__(self, ith_layer, state_dim, lstm_p_dropout,
                 kernel_l2_lambda, recurrent_l2_lambda, bias_l2_lambda, activity_l2_lambda):
        self.retseq_uni_lstm = LSTM(state_dim, return_sequences=True,
                                    name=ith_layer+'th_retseq_uni_lstm',
                                    kernel_regularizer=regularizers.l2(kernel_l2_lambda),
                                    recurrent_regularizer=regularizers.l2(recurrent_l2_lambda),
                                    bias_regularizer=regularizers.l2(bias_l2_lambda),
                                    activity_regularizer=regularizers.l2(activity_l2_lambda))
        # self.retseq_bilstm = Bidirectional(retseq_lstm, merge_mode='concat')
        self.dropout = Dropout(lstm_p_dropout, name=ith_layer+'th_dropout')
        self.layer_norm = LayerNormalization(name=ith_layer+'th_LN')

    def __call__(self, enc_input):
        hidden_seq = self.retseq_uni_lstm(enc_input)
        hidden_seq = self.dropout(hidden_seq)
        # Residual connection/ identical shortcut connection/skip connection
        if enc_input.shape[-1] == hidden_seq.shape[-1]:
            hidden_seq = Add()([hidden_seq, enc_input])
        return self.layer_norm(hidden_seq)


class Decoder:
    def __init__(self, retseq_layer_num, state_dim, p_dropout,
                 kernel_l2_lambda, recurrent_l2_lambda, bias_l2_lambda, activity_l2_lambda):
        self.dec_layers = [DecoderLayer(str(i + 1), state_dim, p_dropout,
                                        kernel_l2_lambda, recurrent_l2_lambda, bias_l2_lambda,
                                        activity_l2_lambda) for i in range(retseq_layer_num)]

    def __call__(self, word_vec_seq):
        x = word_vec_seq
        for dec_layer in self.dec_layers:
            x = dec_layer(x)
        return x


class LRSchedulerPerStep(Callback):
    def __init__(self, n_model, warmup, start_decay, end_decay):
        super(LRSchedulerPerStep, self).__init__()
        self.basic = 1e-4
        self.n_model = n_model
        self.warmup = warmup
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        t = self.step_num
        n = self.n_model
        p = self.warmup
        s = self.start_decay
        e = self.end_decay
        first = 1+t*(n-1)/(n*p)
        second = n
        third = n*(2*n)**((s-n*t)/(e-s))
        lr = self.basic * min(first, second, third)
        K.set_value(self.model.optimizer.lr, lr)
