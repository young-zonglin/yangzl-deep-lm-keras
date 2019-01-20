import sys

from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

from layers import transformer

model_name_abbr_full = {'DULBModel': 'DeepUniLSTMBasedModel',
                        'TEBModel': 'TransformerEncoderBasedModel',
                        'REBModel': "RNMTPlusEncoderBasedModel",
                        'MHABModel': 'MultiHeadAttnBasedModel'}
model_name_full_abbr = {v: k for k, v in model_name_abbr_full.items()}
available_models = ['DULBModel', 'REBModel', 'TEBModel', 'MHABModel']


def get_hyperparams(model_name):
    if model_name == available_models[0]:
        return DeepUniLSTMHParams()
    elif model_name == available_models[1]:
        return RNMTPlusEncoderHParams()
    elif model_name == available_models[2]:
        return TransformerEncoderHParams()
    elif model_name == available_models[3]:
        return MultiHeadAttnHParams()
    else:
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() func, model_name value error.')


# Avoid crossing import between modules.
# Definition need before calling it.
# Calling in a function/method does not require prior definition.
# Class properties will be initialized without instantiation.
class LRSchedulerDoNothing(Callback):
    def __init__(self):
        super(LRSchedulerDoNothing, self).__init__()


# return hyper params string => done
class BasicHParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.mode = 1

        self.time_step = 20
        self.keep_word_num = 10000
        self.word_vec_dim = 300

        self.oov_tag = '<UNK>'
        self.char_level = False
        self.filters = ''

        self.p_dropout = 0.5

        self.batch_size = 128  # Integer multiple of 32

        self.optimizer = Adam()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        self.early_stop_monitor = 'val_loss'
        self.early_stop_mode = 'auto'
        # 10 times waiting is not enough.
        # Maybe 20 is a good value.
        self.early_stop_patience = 20
        self.early_stop_min_delta = 1e-4

        self.train_epoch_times = 1000

        self.pad = 'pre'
        self.cut = 'pre'

    def __str__(self):
        ret_info = list()
        ret_info.append('optimizer: ' + str(self.optimizer) + '\n')
        ret_info.append('lr scheduler: ' + str(self.lr_scheduler) + '\n\n')

        ret_info.append('net architecture mode: ' + str(self.mode) + '\n\n')

        ret_info.append('time step: ' + str(self.time_step) + '\n')
        ret_info.append('keep word num: ' + str(self.keep_word_num) + '\n')
        ret_info.append('word vec dim: ' + str(self.word_vec_dim) + '\n\n')

        ret_info.append('oov tag: ' + self.oov_tag + '\n')
        ret_info.append('char level: ' + str(self.char_level) + '\n')
        ret_info.append('filters: ' + self.filters + '\n\n')

        ret_info.append('dropout proba: ' + str(self.p_dropout) + '\n\n')

        ret_info.append('batch size: '+str(self.batch_size)+'\n\n')

        ret_info.append('kernel l2 lambda: ' + str(self.kernel_l2_lambda) + '\n')
        ret_info.append('recurrent l2 lambda: ' + str(self.recurrent_l2_lambda) + '\n')
        ret_info.append('bias l2 lambda: ' + str(self.bias_l2_lambda) + '\n')
        ret_info.append('activity l2 lambda: ' + str(self.activity_l2_lambda) + '\n\n')

        ret_info.append('early stop monitor: ' + str(self.early_stop_monitor) + '\n')
        ret_info.append('early stop mode: ' + str(self.early_stop_mode) + '\n')
        ret_info.append('early stop patience: ' + str(self.early_stop_patience) + '\n')
        ret_info.append('early stop min delta: ' + str(self.early_stop_min_delta) + '\n\n')

        ret_info.append('train epoch times: ' + str(self.train_epoch_times) + '\n\n')

        ret_info.append("pad: " + self.pad + '\n')
        ret_info.append("cut: " + self.cut + '\n')
        return ''.join(ret_info)


class DeepUniLSTMHParams(BasicHParams):
    """
    The best result is a val_accuracy of about xx.xx%.
    """
    def __init__(self):
        super(DeepUniLSTMHParams, self).__init__()
        self.mode = 1
        self.oov_tag = '<unk>'

        self.unilstm_retseq_layer_num = 1
        self.state_dim = self.word_vec_dim

        self.p_dropout = 0.0

        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        self.optimizer = RMSprop()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.early_stop_monitor = 'val_loss'
        self.early_stop_patience = 40

        self.batch_size = 512

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('uni-lstm retseq layer num: ' + str(self.unilstm_retseq_layer_num) + '\n')
        ret_info.append('state dim: ' + str(self.state_dim) + '\n\n')

        super_str = super(DeepUniLSTMHParams, self).__str__()
        return ''.join(ret_info) + super_str


# The scale of the model and cell state dim should be proportional to the scale of the data.
class RNMTPlusEncoderHParams(BasicHParams):
    def __init__(self):
        super(RNMTPlusEncoderHParams, self).__init__()
        # follow SBLDModel
        self.retseq_layer_num = 1
        self.state_dim = self.word_vec_dim

        # Layer Norm also has a regularization effect
        self.p_dropout = 0.0

        # follow origin paper
        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        self.lr = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-6
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2, epsilon=self.eps)  # follow origin paper
        self.lr_scheduler = LRSchedulerDoNothing()

        self.early_stop_monitor = 'val_acc'

        self.batch_size = 128  # Recommended by "Exploring the Limits of Language Modeling".

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('ret seq layer num: ' + str(self.retseq_layer_num) + '\n')
        ret_info.append('state dim: ' + str(self.state_dim) + '\n\n')

        ret_info.append('lr: ' + str(self.lr) + '\n')
        ret_info.append('beta_1: ' + str(self.beta_1) + '\n')
        ret_info.append('beta_2: ' + str(self.beta_2) + '\n')
        ret_info.append('epsilon: ' + str(self.eps) + '\n')

        super_str = super(RNMTPlusEncoderHParams, self).__str__()
        return ''.join(ret_info) + super_str


class TransformerEncoderHParams(BasicHParams):
    """
    A model configured with the following parameters
    is best able to achieve a val_loss value of approximately 0.xxxx
    or a val_accuracy of about xx.xx%.
    """
    def __init__(self):
        super(TransformerEncoderHParams, self).__init__()
        self.transformer_mode = 0
        self.word_vec_dim = 300
        self.layers_num = 1  # learning ability is enough
        self.d_model = self.word_vec_dim
        # d_ff, imitate original paper
        self.d_inner_hid = int(self.d_model*4)
        self.n_head = 5  # h head, imitate the original paper
        # In original paper, d_k = d_v = 64
        self.d_k = self.d_v = int(self.d_model/self.n_head)
        self.d_pos_enc = self.d_model
        # 0.3 or 0.4 is a good value.
        self.p_dropout = 0.0

        self.lr = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.eps = 1e-9
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2, epsilon=self.eps)  # follow origin paper
        self.warmup_step = 4000  # in origin paper, this value is set to 4000
        # This learning rate scheduling strategy is very important.
        self.lr_scheduler = transformer.LRSchedulerPerStep(self.d_model, self.warmup_step)

        self.pad = 'post'
        self.cut = 'post'

        # acc and val_acc have a stagnant phase at the beginning.
        # loss and val_loss are still declining.
        self.early_stop_monitor = 'val_loss'
        # dropout rate up, stagnation duration also up.
        self.early_stop_patience = 40

        # if set too small => GPU usage rate is low => training is slow
        # if set too large => will miss more samples
        # When set batch size to 512, the performance of the model on the validation set
        # is close to setting the batch size to 128.
        self.batch_size = 512

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('transformer mode: ' + str(self.transformer_mode) + '\n')
        ret_info.append('encoder layer num: ' + str(self.layers_num) + '\n')
        ret_info.append('d_model: ' + str(self.d_model) + '\n')
        ret_info.append('dim of inner hid: ' + str(self.d_inner_hid) + '\n')
        ret_info.append('n head: ' + str(self.n_head) + '\n')
        ret_info.append('dim of k: ' + str(self.d_k) + '\n')
        ret_info.append('dim of v: ' + str(self.d_v) + '\n')
        ret_info.append('pos enc dim: ' + str(self.d_pos_enc) + '\n\n')

        ret_info.append('lr: ' + str(self.lr) + '\n')
        ret_info.append('beta_1: ' + str(self.beta_1) + '\n')
        ret_info.append('beta_2: ' + str(self.beta_2) + '\n')
        ret_info.append('epsilon: ' + str(self.eps) + '\n')
        ret_info.append('warm up step: ' + str(self.warmup_step) + '\n')

        super_str = super(TransformerEncoderHParams, self).__str__()
        return ''.join(ret_info) + super_str


class MultiHeadAttnHParams(BasicHParams):
    def __init__(self):
        super(MultiHeadAttnHParams, self).__init__()
        self.word_vec_dim = 300
        self.d_model = self.word_vec_dim
        self.n_head = 5  # h head
        self.d_k = self.d_v = int(self.d_model / self.n_head)
        self.p_dropout = 0.1

        self.pad = 'post'
        self.cut = 'post'

        self.early_stop_monitor = 'val_loss'

        self.batch_size = 128

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== ' + self.current_classname + ' ==================\n')
        ret_info.append('n head: ' + str(self.n_head) + '\n')
        ret_info.append('dim of k: ' + str(self.d_k) + '\n')
        ret_info.append('dim of v: ' + str(self.d_v) + '\n\n')

        super_str = super(MultiHeadAttnHParams, self).__str__()
        return ''.join(ret_info) + super_str


if __name__ == '__main__':
    print(BasicHParams())
    print(DeepUniLSTMHParams())
    print(RNMTPlusEncoderHParams())
    print(TransformerEncoderHParams())
    print(MultiHeadAttnHParams())
