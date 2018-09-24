import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
# from tensorflow.contrib.crf import CrfDecodeBackwardRnnCell
from general_lveg_test import lveg

from neuronlp2.nn.modules.crf import ChainCRF

# bi-lstm test
np.random.seed(10)
tf.set_random_seed(3)

BATCH = 2
# dynamic length
LEN = 4  # np.random.randint(2, 10)
LABELS = 2
WORD_SIZE = 2
DIM = 10
# alert CRF class
# class BigramCrfForwardRnnCell(rnn_cell.RNNCell):
#     """Computes the alpha values in a linear-chain CRF.
#   See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
#   """
#
#     def __init__(self, transition_params):
#         """Initialize the CrfForwardRnnCell.
#     Args:
#       transition_params: A [batch, length, num_tags, num_tags] matrix of binary potentials.
#           This matrix is expanded into a [1, num_tags, num_tags] in preparation
#           for the broadcast summation occurring within the cell.
#     """
#         self._transition_params = tf.transpose(transition_params, [1, 0, 2, 3])
#         self._num_tags = transition_params.get_shape()[2].value
#         self._position = 0
#
#     @property
#     def state_size(self):
#         return self._num_tags
#
#     @property
#     def output_size(self):
#         return self._num_tags
#
#     def __call__(self, inputs, state, scope=None):
#         """Build the CrfForwardRnnCell.
#     Args:
#       inputs: A [batch_size, num_tags] matrix of unary potentials.
#       state: A [batch_size, num_tags] matrix containing the previous alpha
#           values.
#       scope: Unused variable scope of this cell.
#     Returns:
#       new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
#           values containing the new alpha values.
#     """
#         state = tf.expand_dims(state, 2)
#
#         # This addition op broadcasts self._transitions_params along the zeroth
#         # dimension and state along the second dimension. This performs the
#         # multiplication of previous alpha values and the current binary potentials
#         # in log space.
#         transition_scores = state + self._transition_params[self._position]
#         new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
#         self._position += 1
#         # Both the state and the output of this RNN cell contain the alphas values.
#         # The output value is currently unused and simply satisfies the RNN API.
#         # This could be useful in the future if we need to compute marginal
#         # probabilities, which would require the accumulated alpha values at every
#         # time step.
#         return new_alphas, new_alphas
#
#
# def crf_log_norm(inputs, sequence_lengths, transition_params):
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
#     first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
#     first_input = tf.squeeze(first_input, [1])
#
#     # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
#     # the "initial state" (the unary potentials).
#     def _single_seq_fn():
#         return tf.reduce_logsumexp(first_input, [1])
#
#     def _multi_seq_fn():
#         """Forward computation of alpha values."""
#         rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
#
#         # Compute the alpha values in the forward algorithm in order to get the
#         # partition function.
#         forward_cell = BigramCrfForwardRnnCell(transition_params)
#         _, alphas = rnn.dynamic_rnn(
#             cell=forward_cell,
#             inputs=rest_of_input,
#             sequence_length=sequence_lengths - 1,
#             initial_state=first_input,
#             dtype=tf.float32)
#         log_norm = tf.reduce_logsumexp(alphas, [1])
#         return log_norm
#
#     max_seq_len = tf.shape(inputs)[1]
#     return tf.cond(pred=tf.equal(max_seq_len, 1),
#                    true_fn=_single_seq_fn,
#                    false_fn=_multi_seq_fn)
#
#
# class BigramCrfDecodeForwardRnnCell(rnn_cell.RNNCell):
#     """Computes the forward decoding in a linear-chain CRF.
#   """
#
#     def __init__(self, transition_params):
#         # alert not forget debug CrfDecodeForwardRnnCell for num_tag
#         """Initialize the CrfDecodeForwardRnnCell.
#     Args:
#       transition_params: A [batch, length, num_tags, num_tags] matrix of binary
#         summation occurring within the cell.
#     """
#         self._transition_params = tf.transpose(transition_params, [1, 0, 2, 3])
#         self._num_tags = transition_params.get_shape()[2].value
#         self.position = 0
#
#     @property
#     def state_size(self):
#         return self._num_tags
#
#     @property
#     def output_size(self):
#         return self._num_tags
#
#     def __call__(self, inputs, state, scope=None):
#         """Build the CrfDecodeForwardRnnCell.
#     Args:
#       inputs: A [batch_size, num_tags] matrix of unary potentials.
#       state: A [batch_size, num_tags] matrix containing the previous step's
#             score values.
#       scope: Unused variable scope of this cell.
#     Returns:
#       backpointers: A [batch_size, num_tags] matrix of backpointers.
#       new_state: A [batch_size, num_tags] matrix of new score values.
#     """
#         # For simplicity, in shape comments, denote:
#         # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
#         state = tf.expand_dims(state, 2)  # [B, O, 1]
#
#         # This addition op broadcasts self._transitions_params along the zeroth
#         # dimension and state along the second dimension.
#         # [B, O, 1] + [1, O, O] -> [B, O, O]
#         transition_scores = state + self._transition_params[self.position]  # [B, O, O]
#         new_state = inputs + tf.reduce_max(transition_scores, [1])  # [B, O]
#         backpointers = tf.argmax(transition_scores, 1)
#         backpointers = tf.cast(backpointers, dtype=tf.int32)  # [B, O]
#         self.position += 1
#         return backpointers, new_state
#
#
# class BigramCrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
#     """Computes backward decoding in a linear-chain CRF.
#        Samw with CrfDecodeBackwardRnnCell
#     """
#
#     def __init__(self, num_tags):
#         """Initialize the CrfDecodeBackwardRnnCell.
#     Args:
#       num_tags: An integer. The number of tags.
#     """
#         self._num_tags = num_tags
#
#     @property
#     def state_size(self):
#         return 1
#
#     @property
#     def output_size(self):
#         return 1
#
#     def __call__(self, inputs, state, scope=None):
#         """Build the CrfDecodeBackwardRnnCell.
#     Args:
#       inputs: A [batch_size, num_tags] matrix of
#             backpointer of next step (in time order).
#       state: A [batch_size, 1] matrix of tag index of next step.
#       scope: Unused variable scope of this cell.
#     Returns:
#       new_tags, new_tags: A pair of [batch_size, num_tags]
#         tensors containing the new tag indices.
#     """
#         state = tf.squeeze(state, axis=[1])  # [B]
#         batch_size = tf.shape(inputs)[0]
#         b_indices = tf.range(batch_size)  # [B]
#         indices = tf.stack([b_indices, state], axis=1)  # [B, 2]
#         new_tags = tf.expand_dims(
#             tf.gather_nd(inputs, indices),  # [B]
#             axis=-1)  # [B, 1]
#
#         return new_tags, new_tags
#
#
# def crf_decode(potentials, transition_params, sequence_length):
#     """Decode the highest scoring sequence of tags in TensorFlow.
#   This is a function for tensor.
#   Args:
#     potentials: A [batch_size, max_seq_len, num_tags] tensor of
#               unary potentials.
#     transition_params: A [batch, length, num_tags, num_tags] matrix of
#               binary potentials.
#     sequence_length: A [batch_size] vector of true sequence lengths.
#   Returns:
#     decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
#                 Contains the highest scoring tag indices.
#     best_score: A [batch_size] vector, containing the score of `decode_tags`.
#   """
#
#     # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
#     # and the max activation.
#     def _single_seq_fn():
#         squeezed_potentials = tf.squeeze(potentials, [1])
#         decode_tags = tf.expand_dims(
#             tf.argmax(squeezed_potentials, axis=1), 1)
#         best_score = tf.reduce_max(squeezed_potentials, axis=1)
#         return tf.cast(decode_tags, dtype=tf.int32), best_score
#
#     def _multi_seq_fn():
#         """Decoding of highest scoring sequence."""
#
#         # For simplicity, in shape comments, denote:
#         # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
#         num_tags = potentials.get_shape()[2].value
#
#         # Computes forward decoding. Get last score and backpointers.
#         crf_fwd_cell = BigramCrfDecodeForwardRnnCell(transition_params)
#         initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
#         initial_state = tf.squeeze(initial_state, axis=[1])  # [B, O]
#         inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
#         backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
#             crf_fwd_cell,
#             inputs=inputs,
#             sequence_length=sequence_length - 1,
#             initial_state=initial_state,
#             time_major=False,
#             dtype=tf.int32)
#         backpointers = tf.reverse_sequence(  # [B, T - 1, O]
#             backpointers, sequence_length - 1, seq_dim=1)
#
#         # Computes backward decoding. Extract tag indices from backpointers.
#         crf_bwd_cell = BigramCrfDecodeBackwardRnnCell(num_tags)
#         initial_state = tf.cast(tf.argmax(last_score, axis=1),  # [B]
#                                       dtype=tf.int32)
#         initial_state = tf.expand_dims(initial_state, axis=-1)  # [B, 1]
#         decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
#             crf_bwd_cell,
#             inputs=backpointers,
#             sequence_length=sequence_length - 1,
#             initial_state=initial_state,
#             time_major=False,
#             dtype=tf.int32)
#         decode_tags = tf.squeeze(decode_tags, axis=[2])  # [B, T - 1]
#         decode_tags = tf.concat([initial_state, decode_tags],  # [B, T]
#                                        axis=1)
#         decode_tags = tf.reverse_sequence(  # [B, T]
#             decode_tags, sequence_length, seq_dim=1)
#
#         best_score = tf.reduce_max(last_score, axis=1)  # [B]
#         return decode_tags, best_score
#
#     from tensorflow.python.layers import utils
#     return utils.smart_cond(
#         pred=tf.equal(
#             potentials.shape[1].value or tf.shape(potentials)[1], 1),
#         true_fn=_single_seq_fn,
#         false_fn=_multi_seq_fn)
# np_a = np.random.rand(labels, labels)
# np_b = np.random.rand(labels, labels, labels)
#
# a = tf.placeholder(tf.float32, shape=(labels, labels))
# b = tf.placeholder(tf.float32, shape=(labels, labels, labels))
#
# a_idx = tf.argmax(a, axis=1)
# a_hot = tf.one_hot(a_idx, axis=1, depth=labels)
# a_hot_extend = tf.expand_dims(a_hot, axis=-1)
# b_max = tf.reduce_sum(tf.multiply(b, a_hot_extend), axis=1)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# feed_dict = {a:np_a,
#              b:np_b}
#
# res = sess.run(b_max, feed_dict=feed_dict)
# np_a_idx = np.argmax(np_a, axis=1)
# b_max = np_b[np.arange(labels), np_a_idx]
#
# print(res - b_max)
# print(np_a_idx)
# print(a_idx.eval(session=sess, feed_dict=feed_dict))
# print(b_max)
# print('-'*10)
# print(np_b)
# print('-'*10)
# print(res)

# alert multiply test
# np_a = np.arange(0, batch)
# np_b = np.arange(0, batch*labels)
# np_b = np_b.reshape(batch, labels)
#
# a = tf.placeholder(tf.float32, shape=(batch))
#                    # initializer=tf.contrib.layers.xavier_initializer())
# b = tf.placeholder(tf.float32, shape=(batch, labels))
#                    # initializer=tf.contrib.layers.xavier_initializer())
# # a1_r = tf.range(batch)
# # a2_r = tf.range(batch)
# # a_test = a[a1_r, a2_r]
# test1 = tf.multiply(a, b)
# test2 = tf.multiply(tf.expand_dims(a, axis=-1), b)
#
# err = test1 - test2
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# feed_dict = {a:np_a,
#              b:np_b}
# res = sess.run(err, feed_dict=feed_dict)
# print(res)
# print('-'*10)
# np_test1 = np.multiply(np_a, np_b)
# np_test2 = np.multiply(np_a.reshape((batch, 1)), np_b)
# print(np_a)
# print(np_b)
# print(np_test1)
# print(np_test2)
# print(np_test1 - np_test2)

# alert slice select
# np_a = np.random.rand(batch, labels, labels)
# np_idx = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
#
# a = tf.placeholder(tf.float32, shape=(batch, labels, labels))
#                    # initializer=tf.contrib.layers.xavier_initializer())
# idx = tf.placeholder(tf.int32, shape=(3, 3))
#
# a_s = tf.gather_nd(a, idx)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# feed_dict = {a:np_a,
#              idx:np_idx}
# res = sess.run(a_s, feed_dict=feed_dict)
# print(res)
#
# print(np_a[0, 0, 0])
# print(np_a[1, 1, 1])
# print(np_a[0, 1, 0])

# alert tensorflow loop test
# i = tf.constant(0, dtype=tf.int32, name='i')
# parameters = tf.get_variable('p', shape=(10),
#                              initializer=tf.contrib.layers.xavier_initializer())
# sum = tf.constant(0.0, dtype=tf.float32, name='sum')
#
# def condition(i, parameters, sum):
#     return tf.less(i, tf.shape(parameters))[0]
#
# def body(i, parameters, sum):
#     sum = tf.add(sum,  parameters[i])
#     return tf.add(i, 1), parameters, sum
#
# res = tf.while_loop(condition, body, (i, parameters, sum))
#
# # comp = tf.less(i, tf.shape(parameters))
# # comp_rank = tf.rank(comp)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# res_np = sess.run(res)
# print(res_np)
# print(np.sum(res_np[1]))


# alert crf sequence test


# np_state = np.random.rand(BATCH, LEN, LABELS)
# np_idx = np.random.randint(0, LABELS, BATCH * LEN)
# np_idx = np_idx.reshape(BATCH, LEN)
np_idx = np.array([[0, 0, 1], [0, 0, 1]])
np_word = np.array([[0, 0, 1], [0, 0, 1]])
# np_length = np.random.randint(2, LEN, BATCH)
np_length = np.array([3, 3])
# state = tf.placeholder(dtype=tf.float32, shape=(BATCH, LEN, LABELS),
#                        name='CRF_State')
# trans = tf.placeholder(dtype=tf.float32, shape=(LABELS, LABELS),
#                        name='CRF_Transition_Matrix')

lr = .01
momentum = 0.9
idx = tf.placeholder(dtype=tf.int32, shape=(BATCH, 3),
                     name='tag_indices')
sent = tf.placeholder(dtype=tf.int32, shape=(BATCH, 3),
                      name='sentence')
length = tf.placeholder(dtype=tf.int32, shape=(BATCH),
                        name='sequence_length')

model = lveg(WORD_SIZE, LABELS, gaussian_dim=2)
loss = model.loss(sent, idx, length)
pred, _ = model.decode(sent, idx, length)

optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=True).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    feed = {idx: np_idx,
            sent: np_word,
            length: np_length}
    res = sess.run([optim, loss], feed_dict=feed)[1]
    print(res)
feed = {idx: np_idx,
            sent: np_word,
            length: np_length}
res = sess.run(pred, feed_dict=feed)
print(res)
# unary_score = tf.contrib.crf.crf_unary_score(idx, length, state)
# binary_score = tf.contrib.crf.crf_binary_score(idx, length, trans)
#
# sequence_score = unary_score + binary_score
#
# # shape [BATCH, LEN, LABEL, 1]
# state_pand = tf.expand_dims(state, axis=3)
#
# trans_mask = tf.reshape(tf.sequence_mask(length - 1, maxlen=LEN,
#                                          dtype=tf.float32), [BATCH, LEN, 1, 1])
#
# trans_masked = tf.reshape(trans, [1, 1, LABELS, LABELS]) * trans_mask
# state_mask = tf.reshape(tf.sequence_mask(length, maxlen=LEN, dtype=tf.float32),
#                         [BATCH, LEN, 1, 1])
# state_pand = state_pand * state_mask
# energy = state_pand + trans_masked
#
# begin_idx = tf.slice(idx, [0, 0], [-1, LEN - 1])
# end_idx = tf.slice(idx, [0, 1], [-1, LEN - 1])
#
# flattened_energy = tf.reshape(energy, [-1])
#
# binary_idx = begin_idx * LABELS + end_idx
#
# flattened_indeices = (tf.expand_dims(tf.range(BATCH) * LEN, 1) + tf.expand_dims(tf.range(LEN - 1), 0)) * LABELS * LABELS
# flattened_indeices += binary_idx
#
# falttened_score = tf.gather(flattened_energy, flattened_indeices)
# tf_mask = tf.sequence_mask(length, maxlen=LEN - 1,
#                            dtype=tf.float32)
# score = tf.reduce_sum(falttened_score, 1)
#
# # start_tag_indices = tf.slice(idx, [0, 0], [-1, LEN-1])
# # end_tag_indeices = tf.slice(idx, [0, 1], [-1, LEN-1])
# #
# # flattened_transition_indices = start_tag_indices * LABELS + end_tag_indeices
# # flattened_transition_params = tf.reshape(trans, [-1])
# # binary_scores = tf.gather(flattened_transition_params, flattened_transition_indices)
# #
# # tf_mask = tf.sequence_mask(length, maxlen=LEN,
# #                            dtype=tf.float32)
# # truncated_masks = tf.slice(tf_mask, [0, 1], [-1, -1])
# # truncated_masks_v2 = tf.slice(tf_mask, [0, 0], [-1, -1])
# # binary_score_v2 = tf.reduce_sum(truncated_masks * binary_scores, 1)
#
# feed_dict = {idx: np_idx,
#              state: np_state,
#              trans: np_trans,
#              length: np_length}
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# np_sequence_score, np_energy, np_flattened_indeices, np_score = sess.run([sequence_score, energy, flattened_indeices,
#                                                                           score],
#                                                                          feed_dict=feed_dict)
# print(np_sequence_score - np_score)
# print('-' * 10)
# print(np_energy)
# print('-' * 10)
# print(np_flattened_indeices)
#
# # sequence score calculate
# print('-' * 10)
#
# # np_unary_score = np_state[0, 0, np_idx[0, 0]] + np_state[0, 1, np_idx[0, 1]]
# # np_binary_score = np_trans[np_idx[0, 0], np_idx[0, 1]]
# # print(np_sequence_score - np_unary_score - np_binary_score)
# print(np_sequence_score)
# print(np_score)
# print('-' * 10)
# print(np_length)
# print('-' * 10)
# print(np_idx)
# print('=' * 10)
# print(np_trans)
# print('=' * 10)
# print(np_state)

# alert log_norm test
# zero_energy = tf.zeros([BATCH, LEN, LABELS, LABELS])
# reshape_trans = tf.reshape(trans, [1, 1, LABELS, LABELS]) + zero_energy

# log_norm_v2 = crf_log_norm(state, length, reshape_trans)
# log_norm_v1 = tf.contrib.crf.crf_log_norm(state, length, trans)

# alert decode test
# decode_v1 = tf.contrib.crf.crf_decode(state, trans, length)
# decode_v2 = crf_decode(state, reshape_trans, length)
# alert debug our crf
# crf_v1 = ChainCRF(DIM, LABELS, bigram=False)

# crf_input = tf.placeholder(dtype=tf.float32, shape=(BATCH, LEN, DIM),
#                            name='crf_input')
#
# crf_v2 = ChainCRF(DIM, LABELS, bigram=True)
# loss, sequence_score, log_norm, tran, state = crf_v2.loss(crf_input, idx, lengths=length)
# optim = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9,
#                                    use_nesterov=True).minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# feed_dict = {idx: np_idx,
#                  # state: np_state,
#                  # trans: np_trans,
#                  length: np_length,
#                  crf_input:np_crf_input}
#
# v2, np_score, np_norm, np_tran, np_state = sess.run([loss, sequence_score, log_norm, tran, state], feed_dict=feed_dict)
#
# print(v2[0])
# print('-'*10)
# print(np_score)
# print('-'*10)
# print(np_norm)
# print('-'*10)
# print(np_tran)
# print('-'*10)
# print(np_length)
# print(np_idx)
# print('-'*10)
# print(np_state)
# print('='*20)
#
# for i in range(0, 1000):
#     feed_dict = {idx: np_idx,
#                  # state: np_state,
#                  # trans: np_trans,
#                  length: np_length,
#                  crf_input:np_crf_input}
#     _, v2, np_score, np_norm, np_tran, np_state = sess.run([optim, loss, sequence_score, log_norm, tran, state], feed_dict=feed_dict)
# print(v2[0])
# print('-'*10)
# print(np_score)
# print('-'*10)
# print(np_norm)
# print('-'*10)
# print(np_tran)
# print('-'*10)
# print(np_length)
# print(np_idx)
# print('-'*10)
# print(np_state)
