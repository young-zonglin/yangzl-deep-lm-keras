from configs import net_conf, params
from configs.net_conf import available_models
from configs.params import available_corpus
from models.model_factory import ModelFactory
from utils import tools


def train():
    model_name = available_models[0]
    language_model = ModelFactory.make_model(model_name)
    hyperparams = net_conf.get_hyperparams(model_name)
    corpus_name = available_corpus[0]
    corpus_params = params.get_corpus_params(corpus_name)
    tools.train_model(language_model, hyperparams, corpus_params)


if __name__ == '__main__':
    train()
