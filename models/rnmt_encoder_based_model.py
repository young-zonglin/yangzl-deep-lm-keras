from keras.layers import Dropout

from layers import RNMT_plus
from models.basic_model import BasicModel


class RNMTPlusEncoderBasedModel(BasicModel):
    def __init__(self):
        super(RNMTPlusEncoderBasedModel, self).__init__()

    def _do_build(self, word_vec_seq, word_id_seq):
        word_vec_seq = Dropout(self.hyperparams.p_dropout, name='input_dropout')(word_vec_seq)

        RNMT_plus_encoder = RNMT_plus.Encoder(self.hyperparams.retseq_layer_num,
                                              self.hyperparams.state_dim,
                                              self.hyperparams.p_dropout,
                                              self.hyperparams.kernel_l2_lambda,
                                              self.hyperparams.recurrent_l2_lambda,
                                              self.hyperparams.bias_l2_lambda,
                                              self.hyperparams.activity_l2_lambda)
        hidden_seq = RNMT_plus_encoder(word_vec_seq)
        return hidden_seq
