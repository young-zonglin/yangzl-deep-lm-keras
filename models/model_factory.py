import sys

from configs.net_conf import available_models
from models.deep_unilstm_based_model import DeepUniLSTMBasedModel
from models.rnmt_decoder_based_model import RNMTPlusDecoderBasedModel


class ModelFactory:
    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == available_models[0]:
            return DeepUniLSTMBasedModel()
        elif model_name == available_models[1]:
            return RNMTPlusDecoderBasedModel()
        else:
            raise ValueError('In '+ModelFactory().__class__.__name__ + '.'+
                             sys._getframe().f_code.co_name +
                             '() func, model_name value error.')
