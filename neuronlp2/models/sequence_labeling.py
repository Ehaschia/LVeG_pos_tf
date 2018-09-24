__author__ = 'max'

import tensorflow as tf
from ..nn.modules.crf import ChainCRF
from ..nn.modules.lveg import GeneralLVeG


class BiRecurrentConv(object):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=0.5, p_out=0.5):

        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.rnn_mode = rnn_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tag_space = tag_space
        self.p_in = p_in
        self.num_labels = num_labels
        self.p_rnn = p_rnn
        self.p_out = p_out
        self.tag_space = tag_space
        self.char_dim = char_dim
        if embedd_word is None:
            self.word_embedd_matrix = tf.get_variable('word_embedding_matrix', [num_words, word_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer())

        else:
            word_embedding_initializer = tf.constant_initializer(embedd_word)
            self.word_embedd_matrix = tf.get_variable('word_embedding_matrix', [num_words, word_dim],
                                                      initializer=word_embedding_initializer)
        if embedd_char is None:
            self.char_embedd_matrix = tf.get_variable('char_embedding_matrix', [num_chars, char_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            char_embedding_initializer = tf.constant_initializer(embedd_char)
            self.char_embedd_matrix = tf.placeholder('char_embedding_matrix', [num_chars, char_dim],
                                                     initializer=char_embedding_initializer)
        self.rnn = None

        if rnn_mode == 'RNN':
            self.rnn = self.RNN
        elif rnn_mode == 'LSTM':
            self.rnn = self.LSTM
        elif rnn_mode == 'GRU':
            self.rnn = self.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

    #     self.initializer = initializer
    #     self.reset_parameters()
    def conv1d(self, inputs, num_filters, kernel_size, padding, name='', dropout=0.0, training=False):
        cnn_res = tf.layers.conv1d(inputs, num_filters, kernel_size, padding='valid',
                                   name='conv1d_' + name, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)
        pool_res = tf.reduce_max(cnn_res, axis=1, name='pooling_layer')
        if dropout > 0.0:
            return tf.layers.dropout(pool_res, rate=dropout, training=training, name='CNN_dropout')
        else:
            return pool_res

    def RNN(self, inputs, mask=None, name='', bidirectional=True):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size, name="RNN_fw_" + name, reuse=True)
        if bidirectional:
            rnn_cell_bw = tf.contrib.rnn.BasicRNNCell(self.hidden_size, name="Cell_bw_" + name, reuse=tf.AUTO_REUSE)
            res, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rnn_cell], [rnn_cell_bw],
                                                                       inputs, scope='BiRNN',
                                                                       dtype=tf.float32)
        else:
            res, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)

        if mask is not None:
            res = tf.multiply(res, tf.expand_dims(mask, dim=2))
        return res

    def LSTM(self, inputs, mask=None, name='', bidirectional=True):
        rnn_cell = tf.contrib.rnn.LSTMCell(self.hidden_size, name="LSTM_fw_" + name, reuse=tf.AUTO_REUSE)
        if bidirectional:
            rnn_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size, name="LSTM_bw_" + name, reuse=tf.AUTO_REUSE)
            res, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rnn_cell], [rnn_cell_bw],
                                                                       inputs, scope='BiLSTM',
                                                                       dtype=tf.float32)
        else:
            res, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)

        if mask is not None:
            res = tf.multiply(res, tf.to_float(tf.expand_dims(mask, axis=2)))
        return res

    def GRU(self, inputs, mask=None, name='', bidirectional=True):
        rnn_cell = tf.contrib.rnn.GRUCell(self.hidden_size, name="GRU_fw_" + name)
        if bidirectional:
            rnn_cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size, name="GRU_bw_" + name)
            res, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rnn_cell], [rnn_cell_bw],
                                                                       inputs, scope='BiGRU',
                                                                       dtype=tf.float32)
        else:
            res, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)

        if mask is not None:
            res = tf.multiply(res, tf.expand_dims(mask, dim=2))
        return res

    def fully_layer(self, inputs, out_size, name=''):
        return tf.contrib.layers.fully_connected(inputs, out_size, scope=name, reuse=tf.AUTO_REUSE)

    def _get_rnn_output(self, input_word, input_char, mask=None, length=None, hx=None, training=False):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        if length is None and mask is not None:
            length = tf.to_int32(tf.reduce_sum(mask, axis=1))  # + 1
        # [batch, length, word_dim]
        if length is not None and mask is None:
            mask = tf.sequence_mask(length, maxlen=tf.shape(input_word)[1],
                                    dtype=tf.float32)

        word = tf.nn.embedding_lookup(self.word_embedd_matrix, input_word,
                                      name='get_word_embedding')
        word = tf.layers.dropout(word, rate=self.p_in, training=training)
        # [batch, length, char_length, char_dim]
        char = tf.nn.embedding_lookup(self.char_embedd_matrix, input_char,
                                      'get_char_embedding')
        char_size = tf.shape(char)
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        # alert remove transpose
        char = tf.reshape(char, [char_size[0] * char_size[1], char_size[2], self.char_dim])
        # put into cnn [batch*length, char_length, char_filters]
        # then put into maxpooling [batch * length, char_filters]
        char = self.conv1d(char, self.num_filters, self.kernel_size, self.kernel_size - 1,
                           name='char_conv', training=training, dropout=self.p_in)
        # reshape to [batch, length, char_filters]
        char = tf.reshape(char, [char_size[0], char_size[1], self.num_filters])
        # concatenate word and char [batch, length, word_dim+char_filter]
        rnn_input = tf.concat([word, char], 2)
        # prepare packed_sequence
        seq_output = self.rnn(rnn_input, mask=mask, name='0', bidirectional=True)

        output = tf.layers.dropout(seq_output, rate=self.p_rnn)
        if self.tag_space != 0:
            # [batch, length, tag_space]
            output = tf.layers.dropout(tf.nn.elu(self.fully_layer(output, self.tag_space, name='dense_layer')),
                                       rate=self.p_out, training=training)

        return output, mask, length

    def forward(self, input_word, input_char, mask=None, length=None, hx=None):
        # output from rnn [batch, length, tag_space]
        output, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)
        return output, mask, length

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # [batch, length, tag_space]
        mask = tf.cast(mask, tf.float32)
        output, mask, length = self.forward(input_word, input_char, mask=mask, length=length, hx=hx)
        # [batch, length, num_labels]
        output = self.fully_layer(output, self.num_labels, name='dense_softmax')
        # preds = [batch, length]
        preds = tf.argmax(output[:, :, leading_symbolic:], axis=2)
        preds += leading_symbolic

        if length is not None and target.size(1) != mask.size(1):
            max_len = tf.reduce_max(length)
            # alert here maybe wrong
            target = target[:, :max_len]
        # TODO Here loss is error, should write NLL loss
        if mask is not None:
            return tf.reduce_sum(
                tf.multiply(tf.losses.log_loss(tf.nn.softmax(output, axis=1), tf.one_hot(target, axis=-1)),
                            mask)) / tf.reduce_sum(mask), tf.reduce_sum(
                tf.multiply(tf.equal(preds, target) * mask)), preds
        else:
            return self.nll_loss(tf.logsoftmax(output), target.view(-1)).sum() / num, \
                   (tf.equal(preds, target).type_as(output)).sum(), preds


class BiRecurrentConvCRF(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=0,
                 embedd_word=None, embedd_char=None, p_in=0.2, p_rnn=(0.5, 0.5), p_out=0.5, bigram=False):
        super(BiRecurrentConvCRF, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                 rnn_mode, hidden_size, num_layers, num_labels, tag_space=tag_space,
                                                 embedd_word=embedd_word, embedd_char=embedd_char,
                                                 p_in=p_in, p_rnn=p_rnn, p_out=0.5)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.crf = ChainCRF(out_dim, num_labels, bigram=bigram)

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, mask, length = self._get_rnn_output(input_word, input_char, mask=mask,
                                                    length=length, hx=hx, training=True)

        # if length is not None:
        #     max_len = tf.reduce_max(length)
        #     target = target[:, :max_len]
        #
        # [batch, length, num_label,  num_label]
        loss = self.crf.loss(output, target, mask=mask, lengths=length)
        return tf.reduce_mean(loss)

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, mask, length = self._get_rnn_output(input_word, input_char, mask=mask,
                                                    length=length, hx=hx, training=False)

        if target is None:
            return self.crf.decode(output, mask=mask, leading_symbolic=leading_symbolic), None

        # if length is not None:
        #     max_len = tf.reduce_max(length)
        #     target = target[:, :max_len]

        preds, corr = self.crf.decode(output, mask=mask, target=target, lengths=length,
                                      leading_symbolic=leading_symbolic)
        return preds, corr


class BiRecurrentConvLVeG(BiRecurrentConv):
    def __init__(self, word_dim, num_words, char_dim, num_chars, num_filters, kernel_size, rnn_mode, hidden_size,
                 num_layers, num_labels, tag_space=0, embedd_word=None, embedd_char=None, p_in=0.33,
                 p_rnn=(0.5, 0.5), p_out=0.5, bigram=False, spherical=False, t_comp=1, e_comp=1, gaussian_dim=1,
                 clip=1.0):
        super(BiRecurrentConvLVeG, self).__init__(word_dim, num_words, char_dim, num_chars, num_filters, kernel_size,
                                                  rnn_mode, hidden_size, num_layers, num_labels,
                                                  tag_space=tag_space, embedd_word=embedd_word, embedd_char=embedd_char,
                                                  p_in=p_in, p_rnn=p_rnn)

        out_dim = tag_space if tag_space else hidden_size * 2
        self.lveg = GeneralLVeG(out_dim, num_labels, bigram=bigram, spherical=spherical,
                                gaussian_dim=gaussian_dim, clip=clip)
        self.dense_softmax = None
        self.logsoftmax = None
        self.nll_loss = None

    def loss(self, input_word, input_char, target, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)

        # if length is not None:
        #     max_len = length.max()
        #     target = target[:, :max_len]

        # [batch, length, num_label,  num_label]
        return self.lveg.loss(output, target, length)

    def decode(self, input_word, input_char, target=None, mask=None, length=None, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        output, mask, length = self._get_rnn_output(input_word, input_char, mask=mask, length=length, hx=hx)

        if target is None:
            return self.lveg.decode(output, length, leading_symbolic=leading_symbolic), None

        # if length is not None:
        #     max_len = length.max()
        #     target = target[:, :max_len]

        preds, corr = self.lveg.decode(output, length, target=target, leading_symbolic=leading_symbolic)
        return preds, corr
