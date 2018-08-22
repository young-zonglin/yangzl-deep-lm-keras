from keras import regularizers
from keras.layers import Dropout, LSTM, Bidirectional

from models.basic_model import BasicModel


class DeepBiLSTMBasedModel(BasicModel):
    def __init__(self):
        super(DeepBiLSTMBasedModel, self).__init__()

    def _do_build(self, word_vec_seq, word_id_seq):
        p_dropout = self.hyperparams.p_dropout
        hidden_seq = Dropout(p_dropout, name='input_dropout')(word_vec_seq)

        bilstm_retseq_layer_num = self.hyperparams.bilstm_retseq_layer_num
        state_dim = self.hyperparams.state_dim
        for _ in range(bilstm_retseq_layer_num):
            this_lstm = LSTM(state_dim, return_sequences=True,
                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                             recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            this_bilstm = Bidirectional(this_lstm, merge_mode='concat')
            this_dropout = Dropout(p_dropout)
            hidden_seq = this_bilstm(hidden_seq)
            hidden_seq = this_dropout(hidden_seq)

        return hidden_seq
