import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, 'result')

TRAIN_RECORD_FNAME = 'a.train.info.record'
# 注意：后缀应为png，jpeg之类
MODEL_VIS_FNAME = 'a.model.visual.png'

MATCH_SINGLE_QUOTE_STR = r'[^a-zA-Z]*(\')[a-zA-Z ]*(\')[^a-zA-Z]+'


class DataSetParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.open_file_encoding = 'utf-8'
        self.save_file_encoding = 'utf-8'

        self.raw_path = None
        self.train_url = None
        self.val_url = None
        self.test_url = None

        self.pretrained_word_vecs_url = None

    def __str__(self):
        ret_info = list()
        ret_info.append("open file encoding: " + self.open_file_encoding + '\n')
        ret_info.append("save file encoding: " + self.save_file_encoding + '\n\n')

        ret_info.append("raw path: " + str(self.raw_path) + '\n')
        ret_info.append("train url: " + str(self.train_url) + '\n')
        ret_info.append("val url: " + str(self.val_url) + '\n')
        ret_info.append("test url: " + str(self.test_url) + '\n\n')

        ret_info.append("pretrained word vectors url: " + str(self.pretrained_word_vecs_url) + '\n\n')
        return ''.join(ret_info)


class JustForTest(DataSetParams):
    def __init__(self):
        super(JustForTest, self).__init__()
        self.open_file_encoding = 'gbk'

        # just for test
        just_for_test = os.path.join(PROJECT_ROOT, 'data', 'just_for_test')
        self.raw_path = just_for_test
        self.train_url = just_for_test
        self.val_url = just_for_test
        self.test_url = just_for_test

    def __str__(self):
        ret_info = list()
        ret_info.append('================== ' + self.current_classname + ' ==================\n')

        super_str = super(JustForTest, self).__str__()
        return ''.join(ret_info) + super_str


class SohuNews2008(DataSetParams):
    def __init__(self):
        DataSetParams.__init__(self)
        # raw data
        self.raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'sohu_news_2008', 'raw_data')
        self.fastText_zh_pretrained_wiki_word_vecs_url = os.path.join(self.raw_data_dir,
                                                                      'fast_text_vectors_wiki.zh.vec',
                                                                      'wiki.zh.vec')

        # processed data
        self.processed_data_dir = os.path.join(PROJECT_ROOT, 'data',
                                               'sohu_news_2008', 'processed_data')
        self.processed_zh_word_vecs_url = os.path.join(self.processed_data_dir, 'processed.wiki.zh.vec')

        # raw, train, val, test
        self.raw_path = self.raw_data_dir
        self.train_url = os.path.join(self.processed_data_dir, 'zh_train')
        self.val_url = os.path.join(self.processed_data_dir, 'zh_val')
        self.test_url = os.path.join(self.processed_data_dir, 'zh_test')

        self.pretrained_word_vecs_url = self.processed_zh_word_vecs_url

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        ret_info.append("raw data dir: " + self.raw_data_dir + '\n')
        ret_info.append("processed data dir: " + self.processed_data_dir + '\n\n')

        super_str = super(SohuNews2008, self).__str__()
        return ''.join(ret_info) + super_str


corpus_name_abbr_full = {'just_for_test': JustForTest().__class__.__name__,
                          'sohu_news_2008': SohuNews2008().__class__.__name__}
corpus_name_full_abbr = {v: k for k, v in corpus_name_abbr_full.items()}
available_corpus = ['just_for_test', 'sohu_news_2008']


def get_corpus_params(corpus_name):
    if corpus_name == available_corpus[0]:
        return JustForTest()
    elif corpus_name == available_corpus[1]:
        return SohuNews2008()
    else:
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() func, corpus_name value error.')


if __name__ == '__main__':
    print(DataSetParams())
    print(JustForTest())
    print(SohuNews2008())
