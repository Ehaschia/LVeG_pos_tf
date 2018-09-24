import tensorflow as tf
# from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import rnn_cell
import numpy as np
import math
from crf import BigramCrfForwardRnnCell, BigramCrfDecodeForwardRnnCell, BigramCrfDecodeBackwardRnnCell
from rnn_rewrite import dynamic_bigram_rnn


# from lveg_loop import *

class BasicLVeGCell(rnn_cell.RNNCell):
    def __init__(self, gaussian_dim, num_tags):
        # version 1.0 just 1 component
        # trans_weight [num_tag, num_tag] reshape to [1, num_tag, num_tag]
        # trans_mu [num_tag, num_tag, 2*gaussian_dim] reshape to [1, num_tag, num_tag, 2*gaussian_dim]
        # trans_var [num_tag, num_tag, 2*gaussian_dim, 2*gaussian_dim] reshape to
        # [1, num_tag, num_tag, 2*gaussian_dim, 2*gaussian_dim]
        self._gaussian_dim = gaussian_dim
        self._dim = tf.constant(self._gaussian_dim, dtype=tf.int32)
        self._num_tags = num_tags
        self._tags = tf.constant(self._num_tags, dtype=tf.int32)

    def __call__(self):
        raise ValueError("Basic LVeG Cell, cannot call")

    @property
    def state_size(self):
        raise ValueError("Basic LVeG Cell, cannot use this function")

    @property
    def output_size(self):
        raise ValueError("Basic LVeG Cell, cannot use this function")

    def zeta(self, l, eta, dim, var=None):
        # shape reformat
        # eta_size[-2] = 1
        eta_size = tf.shape(eta)
        eta = tf.expand_dims(eta, axis=-2)
        if var is None:
            var = tf.matrix_inverse(l)
        # if mu is None:
        #     mu = torch.bmm(eta, var)
        # mu = mu.view(-1, 1, dim)
        # can optimize
        # Alert the scale is log format
        # log(2*pi) = 1.8378
        det = tf.log(tf.matrix_determinant(l))
        scale = -0.5 * (dim * 1.837877 - det + tf.squeeze(tf.squeeze(tf.matmul(
            tf.matmul(eta, var), tf.expand_dims(tf.reshape(eta, eta_size), axis=-1)), axis=-1), axis=-1))
        # check_numerics(scale, position="Scale")
        return scale

    def general_gaussian_multi(self, n1_mu, n1_var, n2_mu, n2_var, child, rank):
        dim = self._gaussian_dim

        # calculate lambda, eta and zeta
        # l1 [1, num_tag, num_tag, 2*gaussian_dim, 2*gaussian_dim]
        # eta1 [1, num_tag, num_tag, 2*gaussian_dim]
        l1 = tf.matrix_inverse(n1_var)
        eta1 = tf.squeeze(tf.matmul(l1, tf.expand_dims(n1_mu, axis=-1)), axis=-1)

        # l2 [batch, num_tag, 1, gaussian_dim, gaussian_dim] or
        #    [batch, 1, num_tag, gaussian_dim, gaussian_dim]
        # eta2 [batch, num_tag, 1, gaussian_dim] or
        #      [batch, 1, num_tag, gaussian_dim]
        l2 = tf.matrix_inverse(n2_var)
        eta2 = tf.squeeze(tf.matmul(l2, tf.expand_dims(n2_mu, axis=-1)), axis=-1)

        # zeta1 zata2 size same with l1 and l2 expect gaussian_dim
        zeta1 = self.zeta(l1, eta1, dim=dim*2, var=n1_var)
        zeta2 = self.zeta(l2, eta2, dim=dim, var=n2_var)
        if not child:
            l2_padding = np.zeros([rank, 2])
            l2_padding[-2, 1] = dim
            l2_padding[-1, 1] = dim
            eta2_padding = l2_padding[:-1]
            l2_expand = tf.pad(l2, l2_padding, "CONSTANT")
            # l_multi [batch, num_labels, num_labels, 2*gaussian_dim, 2*gaussian_dim]
            l_multi = l1 + l2_expand

            eta2_expand = tf.pad(eta2, eta2_padding, "CONSTANT")
            # eta_multi [batch, num_labels, num_labels, 2*gaussian_dim]
            eta_multi = eta1 + eta2_expand
            # var_multi [batch, num_labels, num_labels, 2*gaussian_dim, 2*gaussian_dim]
            var_multi = tf.matrix_inverse(l_multi)
            # zeta_multi [batch, num_labels, num_labels]
            zeta_multi = self.zeta(l_multi, eta_multi, dim=dim * 2, var=var_multi)
            # mu_multi [batch, num_labels, num_labels, 2*gaussian_dim]
            mu_multi = tf.squeeze(tf.matmul(var_multi, tf.expand_dims(eta_multi, axis=-1)), axis=-1)
            # mu_res [batch, num_labels, num_labels, gaussian_dim]
            mu_res = tf.split(mu_multi, [dim, dim], axis=-1)[1]
            # var_res [batch, num_labels, num_labels, gaussian_dim, gaussian_dim]
            var_res = tf.split(tf.split(var_multi, [dim, dim], axis=-1)[1], [dim, dim], axis=-2)[1]
        else:
            l2_padding = np.zeros([rank, 2])
            l2_padding[-2, 0] = dim
            l2_padding[-1, 0] = dim
            eta2_padding = l2_padding[:-1]
            # shape is same with not child
            l2_expand = tf.pad(l2, l2_padding, "CONSTANT")
            l_multi = l1 + l2_expand

            eta2_expand = tf.pad(eta2, eta2_padding, "CONSTANT")
            eta_multi = eta1 + eta2_expand

            var_multi = tf.matrix_inverse(l_multi)
            zeta_multi = self.zeta(l_multi, eta_multi, dim=dim * 2, var=var_multi)
            mu_multi = tf.squeeze(tf.matmul(var_multi, tf.expand_dims(eta_multi, axis=-1)), axis=-1)
            mu_res = tf.split(mu_multi, [dim, dim], axis=-1)[0]
            var_res = tf.split(tf.split(var_multi, [dim, dim], axis=-1)[0], [dim, dim], axis=-2)[0]
        scale = zeta1 + zeta2 - zeta_multi
        return scale, mu_res, var_res

    # alert just for diagonal gaussian_dim
    def gaussian_multi_d1(self, n1_mu, n1_var, n2_mu, n2_var):
        var_square_add = n1_var + n2_var
        var_log_square_add = tf.log(var_square_add)

        scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + tf.pow(n1_mu - n2_mu, 2.0) / var_square_add)

        mu = (n1_mu * n2_var + n2_mu * n1_var) / var_square_add

        var = tf.pow(n1_var * n2_var / var_square_add, 0.5)
        scale = tf.reduce_sum(scale, axis=-1)
        return scale, mu, var

    def gaussian_multi(self, n1_mu, n1_var, n2_mu, n2_var):

        l1 = tf.matrix_inverse(n1_var)
        l2 = tf.matrix_inverse(n2_var)

        eta1 = tf.squeeze(tf.matmul(l1, tf.expand_dims(n1_mu, axis=-1)), axis=-1)
        eta2 = tf.squeeze(tf.matmul(l2, tf.expand_dims(n2_mu, axis=-1)), axis=-1)

        zeta1 = self.zeta(l1, eta1, self._gaussian_dim, var=n1_var)
        zeta2 = self.zeta(l2, eta2, self._gaussian_dim, var=n2_var)

        l_sum = l1 + l2
        var_sum = tf.matrix_inverse(l_sum)
        eta_sum = eta1 + eta2
        mu_sum = tf.squeeze(tf.matmul(var_sum, tf.expand_dims(eta_sum, axis=-1)), axis=-1)
        zeta_sum = self.zeta(l_sum, eta_sum, self._gaussian_dim, var=var_sum)

        return (zeta1 + zeta2 - zeta_sum), mu_sum, var_sum

    # only support top 1 max now
    def pruning(self, scale, mu, var):
        # only support shape [batch, num_tag, num_tag] pruning
        batch, num_tag, dim = tf.shape(var)[0], tf.shape(var)[1], tf.shape(var)[-1]
        idx = tf.argmax(scale, axis=1)
        idx_onehot = tf.one_hot(idx, depth=num_tag, axis=1)
        scale_pruned = tf.reduce_sum(idx_onehot*scale, axis=1)
        idx_onehot_mu = tf.expand_dims(idx_onehot, axis=-1)
        mu_pruned = tf.reduce_sum(idx_onehot_mu * mu, axis=1)
        idx_onehot_var = tf.expand_dims(idx_onehot_mu, axis=-1)
        var_pruned = tf.reduce_sum(idx_onehot_var * var, axis=1)
        # alert if pruning scale, it means find a max likelihood sequence
        return scale_pruned, mu_pruned, var_pruned


class LVeGForwardCell(BasicLVeGCell):
    # this cell used for calculate forward score,
    def __init__(self, trans_weight, trans_mu, trans_var, gaussian_dim, num_tags, forward=True):
        # version 1.0 just 1 component
        # trans_weight [num_tag, num_tag] reshape to [1, num_tag, num_tag]
        # trans_mu [num_tag, num_tag, 2*gaussian_dim] reshape to [1, num_tag, num_tag, 2*gaussian_dim]
        # trans_var [num_tag, num_tag, 2*gaussian_dim, 2*gaussian_dim] reshape to
        # [1, num_tag, num_tag, 2*gaussian_dim, 2*gaussian_dim]
        super(LVeGForwardCell, self).__init__(gaussian_dim, num_tags)
        self._trans_weight = tf.expand_dims(trans_weight, 0)
        self._trans_mu = tf.expand_dims(trans_mu, 0)
        self._trans_var = tf.expand_dims(trans_var, 0)
        # self._gaussian_dim = gaussian_dim
        # self._num_tags = tf.shape(self._trans_weight)[-1]
        # self._dim = tf.constant(self._gaussian_dim, dtype=tf.int32)
        self._forward = forward

    @property
    def state_size(self):
        weight_shape = self._num_tags
        mu_shape = tf.constant([self._num_tags, self._gaussian_dim], dtype=tf.int32)
        var_shape = tf.constant([self._num_tags, self._gaussian_dim, self._gaussian_dim], dtype=tf.int32)
        return weight_shape, mu_shape, var_shape

    @property
    def output_size(self):
        weight_shape = self._num_tags
        mu_shape = tf.constant([self._num_tags, self._gaussian_dim], dtype=tf.int32)
        var_shape = tf.constant([self._num_tags, self._gaussian_dim, self._gaussian_dim], dtype=tf.int32)
        return weight_shape, mu_shape, var_shape

    def __call__(self, input, state, scope=None, position=None):
        # here state means previous cell state
        # state_weight [batch, num_tag]
        # state_mu [batch, num_tag, gaussian_dim]
        # state_var [batch, num_tag, gaussian_dim, gaussian_dim]
        # input size is same with state
        state_weight, state_mu, state_var = state
        input_weight, input_mu, input_var = input

        state_mu = tf.expand_dims(state_mu, 2)
        state_var = tf.expand_dims(state_var, 2)
        # scale_p [batch, num_tag, num_tag]
        # mu_p [batch, num_tag, num_tag, gaussian_dim]
        # var_p [batch, num_tag, num_tag, gaussian_dim, gaussian_dim]
        if self._forward:
            scale_p, mu_p, var_p = self.general_gaussian_multi(self._trans_mu, self._trans_var,
                                                               state_mu, state_var, False, 5)
        else:
            scale_p, mu_p, var_p = self.general_gaussian_multi(self._trans_mu, self._trans_var,
                                                               state_mu, state_var, True, 5)
        input_mu = tf.expand_dims(input_mu, 1)
        input_var = tf.expand_dims(input_var, 1)
        # scale_c [batch, num_tag, num_tag, num_tag]
        # mu_c [batch, num_tag, num_tag, num_tag, gaussian_dim]
        # var_c [batch, num_tag, num_tag, num_tag, gaussian_dim, gaussian_dim]
        scale_c, mu_c, var_c = self.gaussian_multi(mu_p, var_p, input_mu, input_var)

        # alert add trans_weight for pruning
        state_weight = tf.expand_dims(state_weight, 2)
        input_weight = tf.expand_dims(input_weight, 1)
        scale = scale_p + scale_c + self._trans_weight + state_weight + input_weight
        scale_pruned, mu_pruned, var_pruned = self.pruning(scale, mu_c, var_c)
        scale_sumed = tf.reduce_logsumexp(scale, axis=1)
        return (scale_sumed , mu_pruned, var_pruned), (scale_sumed , mu_pruned, var_pruned)


class LVeGSequenceScoreCell(BasicLVeGCell):

    def __init__(self, trans_weight, trans_mu, trans_var, gaussian_dim, num_tag):
        super(LVeGSequenceScoreCell, self).__init__(gaussian_dim, num_tag)
        self._trans_weight = tf.transpose(trans_weight, [1, 0])
        self._trans_mu = tf.transpose(trans_mu, [1, 0, 2])
        self._trans_var = tf.transpose(trans_var, [1, 0, 2, 3])

    @property
    def state_size(self):
        weight_shape = tf.constant(1, dtype=tf.int32)
        mu_shape = tf.constant([self._gaussian_dim], dtype=tf.int32)
        var_shape = tf.constant([self._gaussian_dim, self._gaussian_dim], dtype=tf.int32)
        return weight_shape, mu_shape, var_shape

    @property
    def output_size(self):
        weight_shape = tf.constant(1, dtype=tf.int32)
        mu_shape = tf.constant([self._gaussian_dim], dtype=tf.int32)
        var_shape = tf.constant([self._gaussian_dim, self._gaussian_dim], dtype=tf.int32)
        return weight_shape, mu_shape, var_shape

    def __call__(self, input, state, scope=None, position=None):
        if position is None:
            raise ValueError("Should Input position to caluclate sequence score.")
        # [batch]
        a_trans_weight = self._trans_weight[position]
        a_trans_mu = self._trans_mu[position]
        a_trans_var = self._trans_var[position]
        # state weight [batch]
        # input weight [batch, num_tag]
        state_weight, state_mu, state_var = state
        input_weight, input_mu, input_var = input

        state_weight = tf.squeeze(state_weight, -1)
        input_weight = tf.squeeze(input_weight, -1)
        scale_c, mu_c, var_c = self.general_gaussian_multi(a_trans_mu, a_trans_var, state_mu, state_var, False, 3)
        scale_p, mu_p, var_p = self.gaussian_multi(mu_c, var_c, input_mu, input_var)
        scale_p += scale_c + a_trans_weight + state_weight + input_weight
        scale_p = tf.expand_dims(scale_p, -1)
        return (scale_p, mu_p, var_p), (scale_p, mu_p, var_p)


# def lveg_log_norm(self, states, sequence_lengths, transition_params):
#     """Computes the normalization for a CRF.
#   Args:
#     inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
#         to use as input to the CRF layer.
#     sequence_lengths: A [batch_size] vector of true sequence lengths.
#     transition_params: A [num_tags, num_tags] transition matrix.
#   Returns:
#     log_norm: A [batch_size] vector of normalizers for a CRF.
#   """
#     # Split up the first and rest of the inputs in preparation for the forward
#     # algorithm.
#     first_input = tf.slice(states, [0, 0, 0], [-1, 1, -1])
#     first_input = tf.squeeze(first_input, [1])
#
#     # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
#     # the "initial state" (the unary potentials).
#     def _single_seq_fn():
#         return tf.reduce_logsumexp(first_input, [1])
#
#     def _multi_seq_fn():
#         """Forward computation of alpha values."""
#         rest_of_input = tf.slice(states, [0, 1, 0], [-1, -1, -1])
#
#         # Compute the alpha values in the forward algorithm in order to get the
#         # partition function.
#         forward_cell = BigramCrfForwardRnnCell(transition_params)
#         _, alphas = dynamic_bigram_rnn(
#             cell=forward_cell,
#             inputs=rest_of_input,
#             sequence_length=sequence_lengths - 1,
#             initial_state=first_input,
#             dtype=tf.float32)
#         log_norm = tf.reduce_logsumexp(alphas, [1])
#         return log_norm
#
#     max_seq_len = tf.shape(states)[1]
#     return tf.cond(pred=tf.equal(max_seq_len, 1),
#                    true_fn=_single_seq_fn,
#                    false_fn=_multi_seq_fn)


def bigram_sequence_score(energy, idx):
    # energy shape [batch_size, max_len, num_label, num_label]
    batch, max_len, num_label = tf.shape(energy)[0], tf.shape(energy)[1], tf.shape(energy)[2]
    begin_idx = tf.slice(idx, [0, 0], [-1, max_len - 1])
    end_idx = tf.slice(idx, [0, 1], [-1, max_len - 1])

    flattened_energy = tf.reshape(energy, [-1])

    binary_idx = begin_idx * num_label + end_idx

    flattened_indeices = (tf.expand_dims(tf.range(batch) * max_len, 1) +
                          tf.expand_dims(tf.range(max_len - 1),
                                         0)) * num_label * num_label
    flattened_indeices += binary_idx

    falttened_score = tf.gather(flattened_energy, flattened_indeices)
    score = tf.reduce_sum(falttened_score, 1)
    return score
