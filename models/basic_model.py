import os
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding
from keras.models import Model
from keras.models import load_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dropout, Dense
from keras import regularizers

from configs import params, net_conf
from configs.net_conf import model_name_full_abbr
from configs.params import corpus_name_full_abbr
from utils import tools, reader

# Specify which GPU card to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# TensorFlow显存管理
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# batch size和seq len随意，word vec dim训练和应用时应一致
class BasicModel:
    def __init__(self, is_training=True):
        self.is_training = is_training

        self.hyperparams = None
        self.mode = None
        self.time_step = None
        self.keep_word_num = None
        self.vocab_size = None
        self.word_vec_dim = None
        self.batch_size = None
        self.this_model_save_dir = None

        self.pretrained_word_vecs_fname = None
        self.raw_url = None
        self.train_fname = None
        self.val_fname = None
        self.test_fname = None

        self.model = None
        self.embedding_matrix = None
        self.tokenizer = None

        self.open_encoding = None
        self.save_encoding = None

        self.pad = None
        self.cut = None

        self.total_samples_count = 0
        self.train_samples_count = 0
        self.val_samples_count = 0
        self.test_samples_count = 0

    def setup(self, hyperparams, corpus_params):
        self.pretrained_word_vecs_fname = corpus_params.pretrained_word_vecs_url
        self.raw_url = corpus_params.raw_url
        self.train_fname = corpus_params.train_url
        self.val_fname = corpus_params.val_url
        self.test_fname = corpus_params.test_url
        self.open_encoding = corpus_params.open_file_encoding
        self.save_encoding = corpus_params.save_file_encoding
        reader.split_train_val_test(self.raw_url,
                                    self.train_fname, self.val_fname, self.test_fname,
                                    self.open_encoding, self.save_encoding)

        run_which_model = model_name_full_abbr[self.__class__.__name__]
        corpus_name = corpus_name_full_abbr[corpus_params.__class__.__name__]
        setup_time = tools.get_current_time()
        self.this_model_save_dir = \
            params.RESULT_SAVE_DIR + os.path.sep + \
            run_which_model + '_' + corpus_name + '_' + setup_time
        if not os.path.exists(self.this_model_save_dir):
            os.makedirs(self.this_model_save_dir)

        self.hyperparams = hyperparams
        self.mode = hyperparams.mode
        self.keep_word_num = hyperparams.keep_word_num
        self.word_vec_dim = hyperparams.word_vec_dim
        self.time_step = hyperparams.time_step
        self.batch_size = hyperparams.batch_size
        self.tokenizer = reader.fit_tokenizer(self.raw_url, self.keep_word_num,
                                              hyperparams.filters, hyperparams.oov_tag,
                                              hyperparams.char_level,
                                              self.open_encoding)
        self.vocab_size = len(self.tokenizer.word_index)

        self.pad = self.hyperparams.pad
        self.cut = self.hyperparams.cut

        self.total_samples_count = reader.count_lines(self.raw_url, self.open_encoding)
        self.train_samples_count = reader.count_lines(self.train_fname, self.open_encoding)
        self.val_samples_count = reader.count_lines(self.val_fname, self.open_encoding)
        self.test_samples_count = reader.count_lines(self.test_fname, self.open_encoding)

        record_info = list()
        record_info.append('\n================ In setup ================\n')
        record_info.append('Vocab size: %d\n' % self.vocab_size)
        record_info.append('Total samples count: %d\n' % self.total_samples_count)
        record_info.append('Train samples count: %d\n' % self.train_samples_count)
        record_info.append('Val samples count: %d\n' % self.val_samples_count)
        record_info.append('Test samples count: %d\n' % self.test_samples_count)
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def _do_build(self, word_vec_seq, word_id_seq):
        raise NotImplementedError()

    def build(self):
        """
        define model
        template method pattern
        :return: Model object using the functional API
        """
        word_id_seq = Input(name='input_layer', shape=(self.time_step,), dtype='int32')
        if self.pretrained_word_vecs_fname:
            word2vec = reader.load_pretrained_word_vecs(self.pretrained_word_vecs_fname)
            self.embedding_matrix = reader.get_embedding_matrix(word2id=self.tokenizer.word_index,
                                                                word2vec=word2vec,
                                                                vec_dim=self.word_vec_dim)
            embedding = Embedding(input_dim=self.vocab_size + 1,
                                  output_dim=self.word_vec_dim,
                                  weights=[self.embedding_matrix],
                                  input_length=self.time_step,
                                  name='pretrained_word_embedding',
                                  trainable=False)
        else:
            embedding = Embedding(input_dim=self.vocab_size + 1,
                                  output_dim=self.word_vec_dim,
                                  input_length=self.time_step,
                                  name='trainable_word_embedding')

        word_vec_seq = embedding(word_id_seq)
        # print(embedding.input_shape)

        hidden_seq = self._do_build(word_vec_seq, word_id_seq)
        if self.mode == 0:
            enc_lstm = LSTM(self.word_vec_dim,
                            kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                            recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                            bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                            activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            seq_enc = Bidirectional(enc_lstm, name='enc_bilstm')(hidden_seq)
            seq_enc = Dropout(self.hyperparams.p_dropout, name='enc_dropout')(seq_enc)
            preds = Dense(self.vocab_size+1, activation='softmax', name='output_layer',
                          kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                          bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                          activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                          )(seq_enc)
        else:
            # Apply the same Dense layer instance using same weights to each timestep of input.
            preds = TimeDistributed(Dense(self.vocab_size+1, activation='softmax', name="output_layer",
                                          kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                                          bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                                          activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
                                    )(hidden_seq)

        self.model = Model(inputs=word_id_seq, outputs=preds)

        record_info = list()
        record_info.append('\n================ In build ================\n')
        record_info.append(self.hyperparams.__str__())
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)
        print('\n############### Model summary ##################')
        self.model.summary()

        return self.model

    # TODO 优化算法
    # 动态学习率 => done，在回调中更改学习率
    def compile(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.hyperparams.optimizer,
                           metrics=['accuracy'])

        # Transformer-based model的图太复杂太乱，没有看的必要
        # 不要在IDE中打开，否则会直接OOM
        # model_vis_url = self.this_model_save_dir + os.path.sep + params.MODEL_VIS_FNAME
        # plot_model(self.model, to_file=model_vis_url, show_shapes=True, show_layer_names=True)

    def fit_generator(self):
        train_start = float(time.time())
        early_stopping = EarlyStopping(monitor=self.hyperparams.early_stop_monitor,
                                       patience=self.hyperparams.early_stop_patience,
                                       min_delta=self.hyperparams.early_stop_min_delta,
                                       mode=self.hyperparams.early_stop_mode,
                                       verbose=1)
        # callback_instance.set_model(self.model) => set_model方法由Keras调用
        lr_scheduler = self.hyperparams.lr_scheduler
        save_url = \
            self.this_model_save_dir + os.path.sep + \
            'epoch_{epoch:04d}-{'+self.hyperparams.early_stop_monitor+':.5f}' + '.h5'
        model_saver = ModelCheckpoint(save_url,
                                      monitor=self.hyperparams.early_stop_monitor,
                                      mode=self.hyperparams.early_stop_mode,
                                      save_best_only=True, save_weights_only=True, verbose=1)
        history = self.model.fit_generator(reader.generate_batch_data_file(self.train_fname,
                                                                           self.tokenizer,
                                                                           self.mode,
                                                                           self.time_step,
                                                                           self.batch_size,
                                                                           self.vocab_size,
                                                                           self.pad, self.cut,
                                                                           self.open_encoding),
                                           validation_data=reader.generate_batch_data_file(self.val_fname,
                                                                                           self.tokenizer,
                                                                                           self.mode,
                                                                                           self.time_step,
                                                                                           self.batch_size,
                                                                                           self.vocab_size,
                                                                                           self.pad, self.cut,
                                                                                           self.open_encoding),
                                           validation_steps=self.val_samples_count / self.batch_size,
                                           steps_per_epoch=self.train_samples_count / self.batch_size,
                                           epochs=self.hyperparams.train_epoch_times, verbose=2,
                                           callbacks=[model_saver, lr_scheduler, early_stopping])
        tools.show_save_record(self.this_model_save_dir, history, train_start)

    # TODO 评价指标
    def evaluate_generator(self):
        scores = self.model.evaluate_generator(generator=reader.generate_batch_data_file(self.test_fname,
                                                                                         self.tokenizer,
                                                                                         self.mode,
                                                                                         self.time_step,
                                                                                         self.batch_size,
                                                                                         self.vocab_size,
                                                                                         self.pad, self.cut,
                                                                                         self.open_encoding),
                                               steps=self.test_samples_count / self.batch_size)
        record_info = list()
        record_info.append("\n================== 性能评估 ==================\n")
        record_info.append("%s: %.4f\n" % (self.model.metrics_names[0], scores[0]))
        record_info.append("%s: %.2f%%\n" % (self.model.metrics_names[1], scores[1] * 100))
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def save(self, model_url):
        self.model.save_weights(model_url)
        print("\n================== 保存模型 ==================")
        print('The weights of', self.__class__.__name__, 'has been saved in', model_url)

    def load(self, model_url):
        self.model.load_weights(model_url, by_name=True)
        print("\n================== 加载模型 ==================")
        print('Model\'s weights have been loaded from', model_url)

    def __call__(self, x):
        return self.model(x)
