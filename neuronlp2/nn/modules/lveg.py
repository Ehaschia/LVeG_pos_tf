import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn
from lveg_cell import *
from lveg_loop import *

class GeneralLVeG(object):

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
        s_mu = tf.reshape(self.fc_layer(input, self.num_labels*self.gaussian_dim, name='state_mu_layer'),
                          [batch, max_len, self.num_labels, self.gaussian_dim])

        s_weight = tf.reshape(self.fc_layer(input,  self.num_labels, name='state_weight_layer'),
                              [batch, max_len, self.num_labels])
        # s_weight = self.s_weight_em(input).view(batch, length, 1, self.num_labels, 1).transpose(0, 1)
        # alert should change to triangular
        s_cho = tf.reshape(self.fc_layer(input, self.num_labels*self.gaussian_dim*self.gaussian_dim,
                                         name='state_cho_layer'),
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

    def decode(self, sents, lengths, target=None, leading_symbolic=0):
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
        if target is None:
            return preds, None
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
                backpointers, sequence_length - 1, seq_axis=1)

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
