import sys

sys.path.append(".")
sys.path.append("..")

import tensorflow.nn as nn
from neuronlp2.io import get_logger, conllx_data
import time
import os
from neuronlp2.nn.modules.lveg_cell import *
from neuronlp2.nn.modules.lveg_loop import dynamic_lveg
from tensorflow.python.ops import rnn

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ChainCRF(object):
    def __init__(self, word_size, num_labels, **kwargs):
        '''
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            **kwargs:
        '''
        self.num_labels = num_labels
        self.pad_label_id = num_labels
        self.state = tf.Variable(tf.truncated_normal([word_size, self.num_labels], stddev=0.05),
                                 name='embedding_matrix')
        self.trans_nn = None
        self.trans_matrix = tf.Variable(tf.truncated_normal([self.num_labels, self.num_labels], stddev=0.05),
                                        name='trans_matrix')

    def loss(self, input, target, mask=None, length=None):
        '''
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]
        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''

        out_s = nn.embedding_lookup(self.state, input, name='sequence_states')
        log_score, tran_param = tf.contrib.crf.crf_log_likelihood(out_s, target, length,
                                                                  transition_params=self.trans_matrix)
        return -1.0 * tf.reduce_mean(log_score)

    def decode(self, input, target, mask, lengths, leading_symbolic=0):
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
        Returns: Tensor
            decoding results in shape [batch, length]
        """

        out_s = nn.embedding_lookup(self.state, input, name='sequence_states')
        preds, score = tf.contrib.crf.crf_decode(out_s, self.trans_matrix, lengths)
        if mask is not None:
            return preds, tf.reduce_sum(tf.multiply(tf.to_float(tf.equal(preds, target)),
                                                    tf.to_float(mask)))
        else:
            return preds, tf.reduce_sum(tf.to_float(tf.equal(preds, target)))


class lveg(object):

    def __init__(self, word_size, num_labels, bigram=False, spherical=False,
                 gaussian_dim=1, clip=1.0, k=1):
        """
            Only support e_comp = t_comp = 1 situation
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            spherical: bool
                if apply spherical gaussian
            gaussian_dim: int
                the dimension of gaussian
            clip: double
                clamp all elements into [-clip, clip]
            k: int
                pruning the inside score
        """
        self.word_size = word_size
        self.num_labels = num_labels
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.spherical = spherical
        self.gaussian_dim = gaussian_dim
        self.min_clip = -clip
        self.max_clip = clip
        self.k = k
        # self.length = 140
        # Gaussian for every emission rule
        # weight is log form
        # self.state_nn_weight = nn.Linear(self.input_size, self.num_labels * self.e_comp)
        self.s_weight_em = tf.get_variable('state_weight_embedding', shape=(word_size, self.num_labels),
                                           initializer=tf.contrib.layers.xavier_initializer())
        # every label  is a gaussian_dim dimension gaussian
        self.s_mu_em = tf.get_variable('state_mu_embedding', shape=(word_size, self.num_labels, gaussian_dim),
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.s_var_em = tf.get_variable('state_var_embedding',
                                        shape=(word_size, self.num_labels, gaussian_dim, gaussian_dim),
                                        initializer=tf.contrib.layers.xavier_initializer())
        # weight and var is log form
        if self.bigram:
            # self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels * self.num_labels)
            self.trans_nn_weight = self.fc_layer

            # self.trans_nn_mu = nn.Linear(self.input_size,
            #                              self.num_labels * self.num_labels * 2 * self.gaussian_dim)
            self.trans_nn_mu = self.fc_layer

            # (2*gaussian_dim * 2*gaussian_dim) matrix cholesky decomposition
            # self.trans_nn_var = nn.Linear(self.input_size,
            #                               self.num_labels * self.num_labels * (2*self.gaussian_dim+1)*self.gaussian_dim)
            self.trans_nn_var = self.fc_layer
        else:

            self.trans_mat_weight = tf.get_variable('transition_matrix_weight',
                                                    shape=(self.num_labels, self.num_labels),
                                                    initializer=tf.contrib.layers.xavier_initializer())

            self.trans_mat_mu = tf.get_variable('transition_matrix_mu',
                                                shape=(self.num_labels, self.num_labels, 2 * self.gaussian_dim),
                                                initializer=tf.contrib.layers.xavier_initializer())
            # (2*gaussian_dim * 2*gaussian_dim) matrix cholesky decomposition
            self.trans_mat_var = tf.get_variable('transition_matrix_var',
                                                 shape=(self.num_labels, self.num_labels,
                                                        (2 * self.gaussian_dim + 1) * self.gaussian_dim),
                                                 initializer=tf.contrib.layers.xavier_initializer())

            # self.trans_mat_var = Parameter(
            #     torch.Tensor(self.num_labels, self.num_labels, 2*self.gaussian_dim))

    def fc_layer(self, inputs, out_size, name=''):
        return tf.contrib.layers.fully_connected(inputs, out_size, scope=name, reuse=tf.AUTO_REUSE)

    def softplus_layer(self, tensor):
        sng = tf.sign(tensor)
        abs = tf.math.softplus(tf.abs(tensor))
        return tf.multiply(sng, abs, "after_softplus")

    def cho_multi(self, tensor, shape):
        return tf.matmul(tensor, tf.transpose(tensor, shape))

    def get_state_and_trans(self, input, length=None):
        batch, max_len = tf.shape(input)[0], tf.shape(input)[1]

        # compute out_weight, out_mu, out_var by tensor dot
        #
        # [batch, length, input_size] * [input_size, num_label*gaussian_dim]
        #
        # thus weight should be [batch, length, num_label*gaussian_dim] --> [batch, length, num_label, 1]
        #
        # the mu and var tensor should be [batch, length, num_label*gaussian_dim] -->
        # [batch, length, num_label, 1, gaussian_dim]
        #
        # if the s_var is spherical it should be [batch, length, 1, 1]

        # s_weight shape [batch, length, num_label]

        # s_mu = self.s_mu_em(input).view(batch, length, 1, self.num_labels, self.gaussian_dim).transpose(0, 1)
        s_mu = tf.reshape(tf.nn.embedding_lookup(self.s_mu_em, input, name='state_mu_embedding'),
                          [batch, max_len, self.num_labels, self.gaussian_dim])

        s_weight = tf.reshape(tf.nn.embedding_lookup(self.s_weight_em, input, name='state_weight_embedding'),
                              [batch, max_len, self.num_labels])
        # s_weight = self.s_weight_em(input).view(batch, length, 1, self.num_labels, 1).transpose(0, 1)
        # alert should change to triangular
        s_cho = tf.reshape(tf.nn.embedding_lookup(self.s_var_em, input, name='state_cho_embedding'),
                           [batch, max_len, self.num_labels, self.gaussian_dim, self.gaussian_dim])
        # reset s_var
        s_cho = self.softplus_layer(s_cho)

        s_var = self.cho_multi(s_cho, [0, 1, 2, 4, 3])

        t_cho = tf.reshape(self.trans_mat_var,
                           [self.num_labels, self.num_labels, (2 * self.gaussian_dim + 1) * self.gaussian_dim])
        t_cho = self.softplus_layer(t_cho)

        t_cho = tf.contrib.distributions.fill_triangular(t_cho)
        t_var = self.cho_multi(t_cho, [0, 1, 3, 2])

        return s_weight, s_mu, s_var, self.trans_mat_weight, self.trans_mat_mu, t_var

    def lveg_forward(self, sequence_lengths, states_weight,
                     states_mu, states_var, trans_weight,
                     trans_mu, trans_var, forward=True):
        # Split up the first and rest of the inputs in preparation for the forward
        # algorithm.
        if not forward:
            # reverse parameter
            s_w_to_rnn = tf.reverse_sequence(states_weight, sequence_lengths,
                                             seq_axis=1)
            s_mu_to_rnn = tf.reverse_sequence(states_mu, sequence_lengths, seq_axis=1)
            s_var_to_rnn = tf.reverse_sequence(states_var, sequence_lengths, seq_axis=1)

            t_w_to_rnn = tf.transpose(trans_weight, [1, 0])
        else:
            s_w_to_rnn = states_weight
            s_mu_to_rnn = states_mu
            s_var_to_rnn = states_var
            t_w_to_rnn = trans_weight

        t_mu_to_rnn = trans_mu
        t_var_to_rnn = trans_var

        first_input_weight = tf.slice(s_w_to_rnn, [0, 0, 0], [-1, 1, -1])
        first_input_weight = tf.squeeze(first_input_weight, [1])

        first_input_mu = tf.slice(s_mu_to_rnn, [0, 0, 0, 0], [-1, 1, -1, -1])
        first_input_mu = tf.squeeze(first_input_mu, [1])

        first_input_var = tf.slice(s_var_to_rnn, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1])
        first_input_var = tf.squeeze(first_input_var, [1])

        # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
        # the "initial state" (the unary potentials).
        # def _single_seq_fn():
        #     return t_w_to_rnn
        #
        # def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input_mu = tf.slice(s_mu_to_rnn, [0, 1, 0, 0], [-1, -1, -1, -1])
        rest_of_input_weight = tf.slice(s_w_to_rnn, [0, 1, 0], [-1, -1, -1])
        rest_of_input_var = tf.slice(s_var_to_rnn, [0, 1, 0, 0, 0], [-1, -1, -1, -1, -1])

        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = LVeGForwardCell(t_w_to_rnn, t_mu_to_rnn, t_var_to_rnn,
                                       self.gaussian_dim, self.num_labels)
        forward_scores, end_state = dynamic_lveg(
            cell=forward_cell,
            inputs=(rest_of_input_weight, rest_of_input_mu, rest_of_input_var),
            sequence_length=sequence_lengths - 1,
            initial_state=(first_input_weight, first_input_mu, first_input_var),
            dtype=tf.float32)
        return forward_scores, end_state

        # max_seq_len = tf.shape(states_weight)[1]
        # return tf.cond(pred=tf.equal(max_seq_len, 1),
        #                true_fn=_single_seq_fn,
        #                false_fn=_multi_seq_fn)

    def loss(self, sents, target, lengths):
        max_len = tf.shape(sents)[1]
        s_weight, s_mu, s_var, t_weight, t_mu, t_var = self.get_state_and_trans(sents, lengths)
        forward_scores, end_state = self.lveg_forward(lengths, s_weight, s_mu, s_var,
                                                      t_weight, t_mu, t_var)
        # gaussian_dim
        forward_score = tf.reduce_logsumexp(end_state[0], axis=1)
        g_dim = self.gaussian_dim
        num_tag = tf.shape(t_weight)[0]
        batch = tf.shape(s_weight)[0]
        # select in trans_matrix idx
        begin_idx = tf.slice(target, [0, 0], [-1, max_len - 1])
        end_idx = tf.slice(target, [0, 1], [-1, max_len - 1])
        flattened_trans_weight = tf.reshape(t_weight, [-1])
        flattened_trans_mu = tf.reshape(t_mu, [-1, 2 * g_dim])
        flattened_trans_var = tf.reshape(t_var, [-1, 2 * g_dim, 2 * g_dim])
        flattened_idx = begin_idx * num_tag + end_idx
        selected_trans_weight = tf.gather(flattened_trans_weight, flattened_idx)
        selected_trans_mu = tf.gather(flattened_trans_mu, flattened_idx)
        selected_trans_var = tf.gather(flattened_trans_var, flattened_idx)
        # here selected_trans_var should be [batch, dim, dim]
        # select in stats
        offsets = tf.expand_dims(tf.range(batch) * max_len * num_tag, 1)
        offsets += tf.expand_dims(tf.range(max_len) * num_tag, 0)
        flattened_idx = tf.reshape(offsets + target, [-1])
        flattened_state_weight = tf.reshape(s_weight, [-1])
        flattened_state_mu = tf.reshape(s_mu, [-1, g_dim])
        flattened_state_var = tf.reshape(s_var, [-1, g_dim, g_dim])
        selected_states_weight = tf.reshape(tf.gather(flattened_state_weight, flattened_idx),
                                            [batch, max_len])
        selected_states_mu = tf.reshape(tf.gather(flattened_state_mu, flattened_idx),
                                        [batch, max_len, g_dim])
        selected_states_var = tf.reshape(tf.gather(flattened_state_var,
                                                   flattened_idx),
                                         [batch, max_len, g_dim, g_dim])
        sequence_score = self.lveg_sequence_score(selected_states_weight, selected_states_mu,
                                                  selected_states_var, selected_trans_weight,
                                                  selected_trans_mu, selected_trans_var, lengths)
        return tf.reduce_mean(forward_score - sequence_score)

    def decode(self, sents, target, lengths, leading_symbolic=0):
        s_weight, s_mu, s_var, t_weight, t_mu, t_var = self.get_state_and_trans(sents, lengths)
        forward_scores, forward_end_state = self.lveg_forward(lengths, s_weight, s_mu, s_var,
                                                              t_weight, t_mu, t_var)
        forward_score, forward_mu, forward_var = forward_scores

        forward_first_weight = tf.slice(s_weight, [0, 0, 0], [-1, 1, -1])
        forward_first_mu = tf.slice(s_mu, [0, 0, 0, 0], [-1, 1, -1, -1])
        forward_first_var = tf.slice(s_var, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1])

        forward_score = tf.concat([forward_first_weight, forward_score], axis=1)
        forward_mu = tf.concat([forward_first_mu, forward_mu], axis=1)
        forward_var = tf.concat([forward_first_var, forward_var], axis=1)
        batch, max_len, num_tag = tf.shape(forward_mu)[0], tf.shape(forward_mu)[1], tf.shape(forward_mu)[2]
        mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

        backward_scores, backward_end_state = self.lveg_forward(lengths, s_weight, s_mu, s_var,
                                                                t_weight, t_mu, t_var, forward=False)

        backward_score, backward_mu, backward_var = backward_scores

        rev_s_weight = tf.reverse_sequence(s_weight, lengths, seq_axis=1)
        backward_first_weight = tf.slice(rev_s_weight, [0, 0, 0], [-1, 1, -1])

        rev_s_mu = tf.reverse_sequence(s_mu, lengths, seq_axis=1)
        backward_first_mu = tf.slice(rev_s_mu, [0, 0, 0, 0], [-1, 1, -1, -1])

        rev_s_var = tf.reverse_sequence(s_var, lengths, seq_axis=1)
        backward_first_var = tf.slice(rev_s_var, [0, 0, 0, 0, 0], [-1, 1, -1, -1, -1])

        backward_score = tf.concat([backward_first_weight, backward_score], axis=1)
        backward_mu = tf.concat([backward_first_mu, backward_mu], axis=1)
        backward_var = tf.concat([backward_first_var, backward_var], axis=1)

        backward_score = tf.reverse_sequence(backward_score, lengths, seq_axis=1)
        backward_mu = tf.reverse_sequence(backward_mu, lengths, seq_axis=1)
        backward_var = tf.reverse_sequence(backward_var, lengths, seq_axis=1)

        # fill forward and backward var 0 part as eye matix
        reverse_mask = tf.abs(1 - mask)
        eye_mat = tf.eye(self.gaussian_dim, batch_shape=[batch, max_len, self.num_labels])
        eye_mat = eye_mat * tf.reshape(reverse_mask, [batch, max_len, 1, 1, 1])
        forward_var = forward_var + eye_mat
        backward_var = backward_var + eye_mat

        # saved final forward score
        mask_1 = tf.sequence_mask(lengths - 1, maxlen=max_len, dtype=tf.float32)
        reverse_mask_1 = tf.abs(1 - mask_1)
        final_forward_mask = mask * reverse_mask_1
        final_forward_score = forward_score * tf.reshape(final_forward_mask, [batch, max_len, 1])

        calculate_cell = BasicLVeGCell(self.gaussian_dim, self.num_labels)
        expand_t_weight = tf.expand_dims(tf.expand_dims(t_weight, 0), 0)
        expand_t_mu = tf.expand_dims(tf.expand_dims(t_mu, 0), 0)
        expand_t_var = tf.expand_dims(tf.expand_dims(t_var, 0), 0)

        tmp_score, tmp_mu, tmp_var = calculate_cell.general_gaussian_multi(expand_t_mu, expand_t_var,
                                                                           tf.expand_dims(forward_mu, axis=3),
                                                                           tf.expand_dims(forward_var, axis=3),
                                                                           False, 6)

        rest_backward_score = tf.slice(backward_score, [0, 1, 0], [-1, -1, -1])
        rest_backward_mu = tf.slice(backward_mu, [0, 1, 0, 0], [-1, -1, -1, -1])
        rest_backward_var = tf.slice(backward_var, [0, 1, 0, 0, 0], [-1, -1, -1, -1, -1])
        # here backward_first_weight like that just placeholder, will be replaced

        stacked_backward_score = tf.concat([rest_backward_score, backward_first_weight], axis=1)
        stacked_backward_mu = tf.concat([rest_backward_mu, backward_first_mu], axis=1)
        stacked_backward_var = tf.concat([rest_backward_var, backward_first_var], axis=1)

        stacked_backward_score = tf.expand_dims(stacked_backward_score, 2)
        stacked_backward_mu = tf.expand_dims(stacked_backward_mu, 2)
        stacked_backward_var = tf.expand_dims(stacked_backward_var, 2)

        expected_count, _, _ = calculate_cell.gaussian_multi(stacked_backward_mu, stacked_backward_var,
                                                             tmp_mu, tmp_var)
        expected_count += stacked_backward_score + expand_t_weight + tmp_score
        expected_count = expected_count * tf.reshape(mask_1, [batch, max_len, 1, 1]) + \
                         tf.expand_dims(final_forward_score, axis=3)

        preds, score = self.lveg_decode(tf.zeros([batch, max_len, self.num_labels]), expected_count, lengths)
        return preds, tf.reduce_sum(tf.multiply(tf.to_float(tf.equal(preds, target)),
                                                tf.to_float(mask)))

    def lveg_log_norm(self, states, sequence_lengths, transition_params):
        """Computes the normalization for a CRF.
      Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
            to use as input to the CRF layer.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.
      Returns:
        log_norm: A [batch_size] vector of normalizers for a CRF.
      """
        # Split up the first and rest of the inputs in preparation for the forward
        # algorithm.
        first_input = tf.slice(states, [0, 0, 0], [-1, 1, -1])
        first_input = tf.squeeze(first_input, [1])

        # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
        # the "initial state" (the unary potentials).
        def _single_seq_fn():
            return tf.reduce_logsumexp(first_input, [1])

        def _multi_seq_fn():
            """Forward computation of alpha values."""
            rest_of_input = tf.slice(states, [0, 1, 0], [-1, -1, -1])

            # Compute the alpha values in the forward algorithm in order to get the
            # partition function.
            forward_cell = BigramCrfForwardRnnCell(transition_params)
            _, alphas = dynamic_bigram_rnn(
                cell=forward_cell,
                inputs=rest_of_input,
                sequence_length=sequence_lengths - 1,
                initial_state=first_input,
                dtype=tf.float32)
            log_norm = tf.reduce_logsumexp(alphas, [1])
            return log_norm

        max_seq_len = tf.shape(states)[1]
        return tf.cond(pred=tf.equal(max_seq_len, 1),
                       true_fn=_single_seq_fn,
                       false_fn=_multi_seq_fn)

    def lveg_sequence_score(self, states_weight, states_mu, states_var,
                            trans_weight, trans_mu, trans_var, sequence_lengths):
        # Split up the first and rest of the inputs in preparation for the forward
        # algorithm.
        # batch, max_len = states_weight[0], states_weight[1]
        first_input_weight = tf.slice(states_weight, [0, 0], [-1, 1])

        # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
        # the "initial state" (the unary potentials).
        def _single_seq_fn():
            return first_input_weight

        def _multi_seq_fn():
            """Forward computation of alpha values."""
            first_input_mu = tf.slice(states_mu, [0, 0, 0], [-1, 1, -1])
            first_input_var = tf.slice(states_var, [0, 0, 0, 0], [-1, 1, -1, -1])
            first_input_mu = tf.squeeze(first_input_mu, 1)
            first_input_var = tf.squeeze(first_input_var, 1)
            rest_of_input_weight = tf.slice(states_weight, [0, 1], [-1, -1])
            rest_of_input_mu = tf.slice(states_mu, [0, 1, 0], [-1, -1, -1])
            rest_of_input_var = tf.slice(states_var, [0, 1, 0, 0], [-1, -1, -1, -1])

            # Compute the alpha values in the forward algorithm in order to get the
            # partition function.
            # reshape weight to let it in rnn structure
            # first_input_weight_expand = tf.expand_dims(first_input_weight, -1)
            rest_of_input_weight_expand = tf.expand_dims(rest_of_input_weight, -1)

            forward_cell = LVeGSequenceScoreCell(trans_weight, trans_mu, trans_var,
                                                 self.gaussian_dim, self.num_labels)
            inmediate, sequence_scores = dynamic_lveg(
                cell=forward_cell,
                inputs=(rest_of_input_weight_expand, rest_of_input_mu, rest_of_input_var),
                sequence_length=sequence_lengths - 1,
                initial_state=(first_input_weight, first_input_mu, first_input_var),
                dtype=tf.float32)
            return sequence_scores[0]

        # return _multi_seq_fn()
        max_seq_len = tf.shape(states_weight)[1]
        return tf.cond(pred=tf.equal(max_seq_len, 1),
                       true_fn=_single_seq_fn,
                       false_fn=_multi_seq_fn)

    def lveg_decode(self, potentials, transition_params, sequence_length):
        """Decode the highest scoring sequence of tags in TensorFlow.
        This is a function for tensor.
        Args:
        potentials: A [batch_size, max_seq_len, num_tags] tensor of
                  unary potentials.
        transition_params: A [batch, length, num_tags, num_tags] matrix of
                  binary potentials.
        sequence_length: A [batch_size] vector of true sequence lengths.
      Returns:
        decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                    Contains the highest scoring tag indices.
        best_score: A [batch_size] vector, containing the score of `decode_tags`.
      """

        # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
        # and the max activation.
        def _single_seq_fn():
            squeezed_potentials = tf.squeeze(potentials, [1])
            decode_tags = tf.expand_dims(
                tf.argmax(squeezed_potentials, axis=1), 1)
            best_score = tf.reduce_max(squeezed_potentials, axis=1)
            return tf.cast(decode_tags, dtype=tf.int32), best_score

        def _multi_seq_fn():
            """Decoding of highest scoring sequence."""

            # For simplicity, in shape comments, denote:
            # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
            num_tags = potentials.get_shape()[2].value

            # Computes forward decoding. Get last score and backpointers.
            crf_fwd_cell = BigramCrfDecodeForwardRnnCell(transition_params)
            initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
            initial_state = tf.squeeze(initial_state, axis=[1])  # [B, O]
            inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
            backpointers, last_score = dynamic_bigram_rnn(  # [B, T - 1, O], [B, O]
                crf_fwd_cell,
                inputs=inputs,
                sequence_length=sequence_length - 1,
                initial_state=initial_state,
                time_major=False,
                dtype=tf.int32)
            backpointers = tf.reverse_sequence(  # [B, T - 1, O]
                backpointers, sequence_length - 1, seq_dim=1)

            # Computes backward decoding. Extract tag indices from backpointers.
            crf_bwd_cell = BigramCrfDecodeBackwardRnnCell(num_tags)
            initial_state = tf.cast(tf.argmax(last_score, axis=1),  # [B]
                                    dtype=tf.int32)
            initial_state = tf.expand_dims(initial_state, axis=-1)  # [B, 1]
            decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
                crf_bwd_cell,
                inputs=backpointers,
                sequence_length=sequence_length - 1,
                initial_state=initial_state,
                time_major=False,
                dtype=tf.int32)
            decode_tags = tf.squeeze(decode_tags, axis=[2])  # [B, T - 1]
            decode_tags = tf.concat([initial_state, decode_tags],  # [B, T]
                                    axis=1)
            decode_tags = tf.reverse_sequence(  # [B, T]
                decode_tags, sequence_length, seq_dim=1)

            best_score = tf.reduce_max(last_score, axis=1)  # [B]
            return decode_tags, best_score

        from tensorflow.python.layers import utils
        return utils.smart_cond(
            pred=tf.equal(
                potentials.shape[1].value or tf.shape(potentials)[1], 1),
            true_fn=_single_seq_fn,
            false_fn=_multi_seq_fn)


def natural_data():
    batch_size = 16
    num_epochs = 100
    gaussian_dim = 1
    lr = 0.01
    momentum = 0.9
    gamma = 0.0
    schedule = 5
    decay_rate = 0.05

    # train_path = "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn"
    train_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/en-ud-train.conllu_clean_cnn"
    dev_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/en-ud-dev.conllu_clean_cnn"
    test_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/en-ud-test.conllu_clean_cnn"

    logger = get_logger("POSCRFTagger")
    # load data

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("/home/ehaschia/Code/NeuroNLP2/data/alphabets/uden",
                                                 train_path, data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    # use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_numpy(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                type_alphabet)
    # pw_cnt_map, pp_cnt_map = store_cnt(data_train, word_alphabet, pos_alphabet)

    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_numpy(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_test = conllx_data.read_data_to_numpy(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    # network = ChainCRF(word_alphabet.size(), pos_alphabet.size())
    network = lveg(word_alphabet.size(), pos_alphabet.size(), gaussian_dim=1)
    tf_word = tf.placeholder(tf.int32, shape=[None, None], name='word')
    tf_labels = tf.placeholder(tf.int32, shape=[None, None], name='label')
    tf_masks = tf.placeholder(tf.float32, shape=[None, None], name='mask')
    tf_lengths = tf.placeholder(tf.int32, shape=[None], name='length')
    # Training Part
    loss = network.loss(tf_word, tf_labels, lengths=tf_lengths)
    # Pred Part
    preds, corr = network.decode(tf_word, lengths=tf_lengths, target=tf_labels,
                                 leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

    # network = ChainCRF(word_alphabet.size(), pos_alphabet.size())
    # store_gaussians(network, word_alphabet, pos_alphabet, '0', pw_cnt_map, pp_cnt_map, pre, threshold)

    # optim = torch.optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    # optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)
    optims = []
    tmp_lr = lr
    # for i in range(0, num_epochs+1):
    #     if i % schedule == 0:
    #         tmp_lr = tmp_lr / (1.0 + i * decay_rate)
    #
    #         optims.append(tf.train.MomentumOptimizer(learning_rate=tmp_lr,
    #                                                  momentum=momentum,
    #                                                  use_nesterov=True).minimize(loss))
    optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=True).minimize(loss)
    # optim = optims[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        for batch in range(1, num_batches + 1):
            word, _, labels, _, _, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size,
                                                                                   unk_replace=0.0)

            max_len = np.max(lengths)
            word = word[:, :max_len]
            labels = labels[:, :max_len]
            feed_dict = {tf_word: word,
                         tf_labels: labels,
                         tf_lengths: lengths}
            # loss_batch = sess.run([optim, loss], feed_dict=feed_dict)[1]
            loss_batch = sess.run([optim, loss], feed_dict=feed_dict)[1]
            # store_input(word, labels, masks, batch, epoch)
            # store_data(network, loss, batch, epoch)
            num_inst = word.shape[0]
            train_err += loss_batch * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        # evaluate performance on dev data
        # if epoch % 10 == 1:
        #     store_param(network, epoch, pos_alphabet, word_alphabet)
        # if epoch % 20 == 1:
        #     store_gaussians(network, word_alphabet, pos_alphabet, str(epoch), pw_cnt_map, pp_cnt_map, pre, threshold)

        dev_corr = 0.0
        dev_total = 0
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch

            max_len = np.max(lengths)
            word = word[:, :max_len]
            labels = labels[:, :max_len]
            feed_dict = {tf_word: word,
                         tf_labels: labels,
                         tf_lengths: lengths}
            pred, cor = sess.run([preds, corr], feed_dict=feed_dict)
            num_tokens = np.sum(lengths)
            dev_corr += cor
            dev_total += num_tokens
        print('dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total))

        if dev_correct < dev_corr:
            dev_correct = dev_corr
            best_epoch = epoch

            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_total = 0
            for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                word, char, labels, _, _, masks, lengths = batch

                max_len = np.max(lengths)
                word = word[:, :max_len]
                labels = labels[:, :max_len]
                feed_dict = {tf_word: word,
                             tf_labels: labels,
                             tf_masks: masks,
                             tf_lengths: lengths}
                pred, cor = sess.run([preds, corr], feed_dict=feed_dict)
                num_tokens = np.sum(lengths)
                test_corr += cor
                test_total += num_tokens
            test_correct = test_corr
        print("best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch))
        if test_total != 0:
            print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                test_correct, test_total, test_correct * 100 / test_total, best_epoch))

        # if epoch % schedule == 0:
        #     tmp = epoch / schedule
        #
        #     optim = optims[tmp]
        # optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)



if __name__ == '__main__':
    tf.set_random_seed(48)
    np.random.seed(48)
    natural_data()
    # main()
