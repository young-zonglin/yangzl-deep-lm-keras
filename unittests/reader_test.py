import unittest

import nltk

from configs import params
from configs.params import available_corpus
from utils import reader
from utils import tools


class ReaderTest(unittest.TestCase):
    corpus = available_corpus[0]
    corpus = params.get_corpus_params(corpus)

    def setUp(self):
        print('set up...')

    def tearDown(self):
        print('tear down...')

    @staticmethod
    def test_load_word_vecs():
        word2vec = reader.load_word_vecs(ReaderTest.corpus.fastText_zh_pretrained_wiki_word_vecs_url,
                                         head_n=50)
        for word, vector in word2vec.items():
            print(word, end=' => ')
            for value in vector:
                print(value, end=' ')
            print()

    @staticmethod
    def test_read_words():
        all_distinct_words = reader.read_words(ReaderTest.corpus.raw_url)
        for word in all_distinct_words:
            print(word, end=' | ')
        print('\nTotal distinct words number:', len(all_distinct_words))

    @staticmethod
    def test_get_needed_vectors():
        word2vec = reader.get_needed_vectors(raw_path=ReaderTest.corpus.raw_url,
                                             full_vecs_fname=ReaderTest.corpus.fastText_zh_pretrained_wiki_word_vecs_url,
                                             needed_vecs_fname=ReaderTest.corpus.processed_zh_word_vecs_url)
        for word, vector in word2vec.items():
            print(word, end=' => ')
            print(vector)

    @staticmethod
    def test_NLTK():
        sentence = "i'v     isn't   can't haven't aren't won't i'm it's we're who's where's i'd we'll we've he's."
        tokens = nltk.word_tokenize(tools.remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR).lower())
        print(tokens)
        for token in tokens:
            print(token, end=' ')
        print()

    @staticmethod
    def test_split_train_val_test():
        raw_path = "E:\\test"
        train_fname = 'E:\\test\\train'
        val_fname = 'E:\\test\\val'
        test_fname = 'E:\\test\\test'
        reader.split_train_val_test(raw_path, train_fname, val_fname, test_fname)


if __name__ == '__main__':
    unittest.main()
