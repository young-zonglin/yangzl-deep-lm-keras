import sys

from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

model_name_abbr_full = {'DULBModel': 'DeepUniLSTMBasedModel',
                        'RDBModel': "RNMTPlusDecoderBasedModel"}
model_name_full_abbr = {v: k for k, v in model_name_abbr_full.items()}
available_models = ['DULBModel', 'RDBModel']


def get_hyperparams(model_name):
    if model_name == available_models[0]:
        return DeepUniLSTMHParams()
    elif model_name == available_models[1]:
        return RNMTPlusDecoderHParams()
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

        self.batch_size = 64

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('uni-lstm retseq layer num: ' + str(self.unilstm_retseq_layer_num) + '\n')
        ret_info.append('state dim: ' + str(self.state_dim) + '\n\n')

        super_str = super(DeepUniLSTMHParams, self).__str__()
        return ''.join(ret_info) + super_str


# The scale of the model and cell state dim should be proportional to the scale of the data.
class RNMTPlusDecoderHParams(BasicHParams):
    def __init__(self):
        super(RNMTPlusDecoderHParams, self).__init__()
        # follow DULBModel
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

        super_str = super(RNMTPlusDecoderHParams, self).__str__()
        return ''.join(ret_info) + super_str


if __name__ == '__main__':
    print(BasicHParams())
    print(DeepUniLSTMHParams())
    print(RNMTPlusDecoderHParams())
