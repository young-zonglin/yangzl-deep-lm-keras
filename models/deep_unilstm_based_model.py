from keras import regularizers
from keras.layers import Dropout, LSTM, Bidirectional

from models.basic_model import BasicModel


class DeepUniLSTMBasedModel(BasicModel):
    def __init__(self):
        super(DeepUniLSTMBasedModel, self).__init__()

    def _do_build(self, word_vec_seq, word_id_seq):
        p_dropout = self.hyperparams.p_dropout
        hidden_seq = Dropout(p_dropout, name='input_dropout')(word_vec_seq)

        for i in range(self.hyperparams.unilstm_retseq_layer_num):
            this_lstm = LSTM(self.hyperparams.state_dim, return_sequences=True,
                             name=str(i+1)+'th_retseq_uni_lstm',
                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                             recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            # this_bilstm = Bidirectional(this_lstm, merge_mode='concat')
            this_dropout = Dropout(p_dropout, name=str(i+1)+'th_dropout')
            hidden_seq = this_lstm(hidden_seq)
            hidden_seq = this_dropout(hidden_seq)

        return hidden_seq
