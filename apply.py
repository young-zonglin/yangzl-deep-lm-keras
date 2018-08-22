from configs import params, net_conf
from configs.net_conf import available_models
from configs.params import available_corpus
from models.model_factory import ModelFactory


def apply():
    model_name = available_models[0]
    language_model = ModelFactory.make_model(model_name)
    model_url = ''
    language_model.load(model_url)
    hyperparams = net_conf.get_hyperparams(model_name)
    corpus_name = available_corpus[0]
    corpus_params = params.get_corpus_params(corpus_name)

    language_model.setup(hyperparams, corpus_params)
    language_model.evaluate_generator()


if __name__ == '__main__':
    apply()
