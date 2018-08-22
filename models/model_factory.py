from configs.net_conf import available_models
from models.basic_model import BasicModel
from models.deep_bilstm_based_model import DeepBiLSTMBasedModel


class ModelFactory:
    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == available_models[0]:
            return DeepBiLSTMBasedModel()
        else:
            return BasicModel()
