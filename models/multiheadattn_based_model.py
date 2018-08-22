from keras.layers import Dropout

from layers.bojone_attention_keras import MultiHeadAttn, PositionEncoding
from models.basic_model import BasicModel


class MultiHeadAttnBasedModel(BasicModel):
    def __init__(self):
        super(MultiHeadAttnBasedModel, self).__init__()

    def _do_build(self, word_vec_seq, word_id_seq):
        pos_enc_layer = PositionEncoding(name='pos_enc_layer')
        input_dropout = Dropout(self.hyperparams.p_dropout, name='input_dropout')
        multi_head_attn_layer = MultiHeadAttn(self.hyperparams.n_head,
                                              self.hyperparams.d_k,
                                              name='multi_head_attn_layer')

        emb_seq = pos_enc_layer(word_vec_seq)
        emb_seq = input_dropout(emb_seq)
        seq_repr_seq = multi_head_attn_layer([emb_seq, emb_seq, emb_seq])
        return seq_repr_seq
