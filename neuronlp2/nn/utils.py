import collections
from itertools import repeat
import tensorflow as tf

import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# def prepare_rnn_seq(rnn_input, lengths, hx=None, masks=None, batch_first=False):
#     '''
#
#     Args:
#         rnn_input: [seq_len, batch, input_size]: tensor containing the features of the input sequence.
#         lengths: [batch]: tensor containing the lengthes of the input sequence
#         hx: [num_layers * num_directions, batch, hidden_size]: tensor containing the initial hidden state for each element in the batch.
#         masks: [seq_len, batch]: tensor containing the mask for each element in the batch.
#         batch_first: If True, then the input and output tensors are provided as [batch, seq_len, feature].
#
#     Returns:
#
#     '''
#     def check_decreasing(lengths):
#         lens= np.sort(lengths)[::-1]
#         order = np.argsort(lengths)[::-1]
#         if np.not_equal(lens, lengths).sum() == 0:
#             return None
#         else:
#             rev_order = np.argsort(order)
#             return lens, order, rev_order
#
#     check_res = check_decreasing(lengths)
#
#     if check_res is None:
#         lens = lengths
#         rev_order = None
#     else:
#         lens, order, rev_order = check_res
#         batch_dim = 0 if batch_first else 1
#         rnn_input = tf.gather_nd(batch_dim, order)
#         if hx is not None:
#             # hack lstm
#             if isinstance(hx, tuple):
#                 hx, cx = hx
#                 hx = hx.index_select(1, order)
#                 cx = cx.index_select(1, order)
#                 hx = (hx, cx)
#             else:
#                 hx = hx.index_select(1, order)
#
#     lens = lens.tolist()
#     # alert useless
#     # seq = rnn_utils.pack_padded_sequence(rnn_input, lens, batch_first=batch_first)
#     mask_seq = tf.sequence_mask(masks, lens[0])
#     seq = tf.multiply(rnn_input, mask_seq)
#     if masks is not None:
#         if batch_first:
#             masks = masks[:, :lens[0]]
#         else:
#             masks = masks[:lens[0]]
#     return seq, hx, rev_order, masks


def check_numerics(input):
    tf.check_numerics(input)
    # check_big = 1.0*np.greater(np.abs(input.data.cpu().numpy()), 1e6)
    # if np.sum(check_big) != 0.0:
    #     print("Too big error!")
    #     exit(1)


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda(sequence_length.get_device())
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    # change for debug test_lveg
    lengths = np.sum(lengths, axis=1).astype(int)
    # if lengths.is_cuda:
    #     lengths = torch.sum(lengths, dim=0).cpu().numpy().astype(int)
    # else:
    #     lengths = torch.sum(lengths, dim=0).numpy().astype(int)
    if not batch_first:
        inputs = tf.transpose(inputs, perm=[0, 1])
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(tf.shape(inputs)[1]))
                        for _ in range(inputs.size(0))]

    input_size = inputs.size()
    inputs = inputs.contiguous()
    inputs = inputs.view(input_size[0], input_size[1], -1)

    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    # reversed_indices = Variable(reversed_indices)
    if inputs.is_cuda:
        device = inputs.get_device()
        reversed_indices = reversed_indices.cuda(device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    reversed_inputs = reversed_inputs.view(input_size)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
