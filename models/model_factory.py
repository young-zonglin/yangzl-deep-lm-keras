import sys

from configs.net_conf import available_models
from models.deep_bilstm_based_model import DeepBiLSTMBasedModel
from models.multiheadattn_based_model import MultiHeadAttnBasedModel
from models.rnmt_encoder_based_model import RNMTPlusEncoderBasedModel
from models.transformer_encoder_based_model import TransformerEncoderBasedModel


class ModelFactory:
    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == available_models[0]:
            return DeepBiLSTMBasedModel()
        elif model_name == available_models[1]:
            return TransformerEncoderBasedModel()
        elif model_name == available_models[2]:
            return RNMTPlusEncoderBasedModel()
        elif model_name == available_models[3]:
            return MultiHeadAttnBasedModel()
        else:
            raise ValueError('In '+ModelFactory().__class__.__name__ + '.'+
                             sys._getframe().f_code.co_name +
                             '() func, model_name value error.')
