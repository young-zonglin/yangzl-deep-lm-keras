"""
Find best hyper params of model by grid search.
Then, save them to hyper params configuration class.
"""
from configs import net_conf, params
from configs.net_conf import available_models, model_name_abbr_full
from configs.params import available_corpus
from models.model_factory import ModelFactory
from utils import tools


def tune_dropout_rate_DULBModel():
    model_name = available_models[0]
    model_full_name = model_name_abbr_full[model_name]
    print('============ ' + model_full_name + ' tune dropout rate ============')
    # Don't set dropout rate too large, because it will cause information loss.
    p_dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for p_dropout in p_dropouts:
        language_model = ModelFactory.make_model(model_name)
        hyperparams = net_conf.get_hyperparams(model_name)
        hyperparams.p_dropout = p_dropout
        corpus_name = available_corpus[2]
        corpus_params = params.get_corpus_params(corpus_name)
        tools.train_model(language_model, hyperparams, corpus_params)


if __name__ == '__main__':
    tune_dropout_rate_DULBModel()
