from configs import params, net_conf
from configs.net_conf import available_models
from configs.params import available_corpus
from models.model_factory import ModelFactory
from utils import tools


def main():
    run_this_model = available_models[0]
    language_model = ModelFactory.make_model(run_this_model)
    hyperparams = net_conf.get_hyperparams(run_this_model)
    hyperparams.batch_size = 1
    corpus_name = available_corpus[0]
    corpus_params = params.get_corpus_params(corpus_name)
    tools.train_model(language_model, hyperparams, corpus_params)


if __name__ == '__main__':
    main()
