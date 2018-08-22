from keras.layers import Embedding, Lambda

from layers import transformer
from models.basic_model import BasicModel


class TransformerEncoderBasedModel(BasicModel):
    def __init__(self):
        super(TransformerEncoderBasedModel, self).__init__()

    def _do_build(self, word_vec_seq, word_id_seq):
        d_model = self.hyperparams.d_model
        d_inner_hid = self.hyperparams.d_inner_hid
        n_head = self.hyperparams.n_head
        d_k = d_v = self.hyperparams.d_k
        d_pos_enc = self.hyperparams.d_pos_enc
        len_limit = self.time_step
        layers_num = self.hyperparams.layers_num
        p_dropout = self.hyperparams.p_dropout

        # 位置编号从1开始
        # word id亦从1开始
        pos_enc_layer = Embedding(len_limit + 1, d_pos_enc, trainable=False,
                                  weights=[transformer.get_pos_enc_matrix(len_limit+1, d_pos_enc)],
                                  name='pos_enc_layer')
        transformer_encoder = transformer.Encoder(d_model, d_inner_hid, n_head, d_k, d_v,
                                                  layers_num=layers_num,
                                                  p_dropout=p_dropout,
                                                  pos_enc_layer=pos_enc_layer,
                                                  mode=self.hyperparams.transformer_mode,
                                                  batch_size=self.batch_size)
        pos_seq = Lambda(transformer.get_pos_seq, name='get_pos_seq')(word_id_seq)
        seq_repr_seq = transformer_encoder(word_vec_seq, word_id_seq, pos_seq)
        return seq_repr_seq
