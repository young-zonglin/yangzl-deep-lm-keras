from configs import params, net_conf
from configs.net_conf import available_models
from configs.params import available_corpus
from models.model_factory import ModelFactory


def apply():
    model_name = available_models[1]
    language_model = ModelFactory.make_model(model_name)
    hyperparams = net_conf.get_hyperparams(model_name)
    corpus_name = available_corpus[0]
    corpus_params = params.get_corpus_params(corpus_name)
    model_url = 'E:\PyCharmProjects\yangzl-lm-keras\\result' \
                '\REBModel_just_for_test_2018-10-19 00_38_16\epoch_0014-0.99687.h5'

    hyperparams.batch_size = 1
    language_model.setup(hyperparams, corpus_params)
    language_model.build()
    language_model.load(model_url)
    language_model.compile()
    language_model.evaluate_generator()


if __name__ == '__main__':
    apply()
