import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from rnn_rewrite import dynamic_bigram_rnn


class ChainCRF(object):
    def __init__(self, input_size, num_labels, bigram=True):
        self.input_size = input_size
        self.num_labels = num_labels  # + 1
        self.bigram = bigram
        if not self.bigram:
            self.trans_matrix = tf.get_variable('transition_matrix', [self.num_labels, self.num_labels],
                                                initializer=tf.contrib.layers.xavier_initializer())

    def forward_bigram(self, input, scope=''):
        batch, length = tf.shape(input)[0], tf.shape(input)[1]
        trans_nn = tf.contrib.layers.fully_connected(input, self.num_labels * self.num_labels,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     scope=scope,
                                                     reuse=tf.AUTO_REUSE)
        out_t = tf.reshape(trans_nn, [batch, length, self.num_labels, self.num_labels])
        return out_t

    def bigram_sequence_score(self, state, trans, idx, length, mask):
        # energy shape [batch_size, max_len, num_label, num_label]
        batch, max_len, num_label = tf.shape(state)[0], tf.shape(state)[1], tf.shape(state)[2]
        trans_mask = tf.reshape(tf.sequence_mask(length - 1, maxlen=max_len,
                                                 dtype=tf.float32), [batch, max_len, 1, 1])
        masked_trans = trans * trans_mask
        masked_state = tf.expand_dims(state, 3) * tf.reshape(mask, [batch, max_len, 1, 1])
        energy = masked_state + masked_trans

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

    def bigram_crf_log_norm(self, states, sequence_lengths, transition_params):
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

    def bigram_crf_decode(self, potentials, transition_params, sequence_length):
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

    def loss(self, input, target, mask=None, lengths=None):
        state = tf.contrib.layers.fully_connected(input, self.num_labels,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  scope='input_to_state_layer')
        if mask is None and lengths is not None:
            mask = tf.sequence_mask(lengths, maxlen=tf.shape(input)[1],
                                    dtype=tf.float32)
        if mask is not None and lengths is None:
            lengths = tf.to_float(tf.reduce_sum(mask, 1))
        if self.bigram:
            trans = self.forward_bigram(input, scope='input_to_trans_layer')
            sequence_score = self.bigram_sequence_score(state, trans, target, lengths, mask)

            log_norm = self.bigram_crf_log_norm(state, lengths, trans)
            log_score = sequence_score - log_norm
        else:
            # alert test loss
            log_score, tran_param = tf.contrib.crf.crf_log_likelihood(state, target, lengths,
                                                                      transition_params=self.trans_matrix)
        return -1.0 * log_score

        # shape = [length, batch, num_label, num_label]

    def decode(self, input, target=None, mask=None, lengths=None, leading_symbolic=0):
        state = tf.contrib.layers.fully_connected(input, self.num_labels,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  scope='input_to_state_layer', reuse=tf.AUTO_REUSE)

        if self.bigram:
            trans = self.forward_bigram(input, scope='input_to_trans_layer')
            preds, score = self.bigram_crf_decode(state, trans, lengths)
        else:
            preds, score = tf.contrib.crf.crf_decode(state, self.trans_matrix, lengths)

        if mask is not None:
            return preds, tf.reduce_sum(tf.multiply(tf.to_float(tf.equal(preds, target)),
                                                    tf.to_float(mask)))
        else:
            return preds, tf.reduce_sum(tf.to_float(tf.equal(preds, target)), mask)


class BigramCrfForwardRnnCell(rnn_cell.RNNCell):
    """Computes the alpha values in a bigram linear-chain CRF.
  """

    def __init__(self, transition_params):
        # why not init super class?
        """Initialize the CrfForwardRnnCell.
    Args:
      transition_params: A [batch, length, num_tags, num_tags] matrix of binary potentials.
          for the broadcast summation occurring within the cell.
    """
        self._transition_params = tf.transpose(transition_params, [1, 0, 2, 3])
        self._num_tags = transition_params.get_shape()[2].value
        self._position = tf.constant(0, dtype=tf.int32)

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None, position=None):
        """Build the BigramCrfForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
          values.
      scope: Unused variable scope of this cell.
    Returns:
      new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
          values containing the new alpha values.
    """
        state = tf.expand_dims(state, 2)
        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension. This performs the
        # multiplication of previous alpha values and the current binary potentials
        # in log space.
        if position is None:
            position = tf.constant(0, dtype=tf.int32)

        transition_scores = state + self._transition_params[position]
        new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
        # Both the state and the output of this RNN cell contain the alphas values.
        # The output value is currently unused and simply satisfies the RNN API.
        # This could be useful in the future if we need to compute marginal
        # probabilities, which would require the accumulated alpha values at every
        # time step.
        return new_alphas, new_alphas


class BigramCrfDecodeForwardRnnCell(rnn_cell.RNNCell):
    """Computes the forward decoding in a linear-chain CRF.
  """

    def __init__(self, transition_params):
        # alert not forget debug CrfDecodeForwardRnnCell for num_tag
        """Initialize the CrfDecodeForwardRnnCell.
    Args:
      transition_params: A [batch, length, num_tags, num_tags] matrix of binary
        summation occurring within the cell.
    """
        self._transition_params = tf.transpose(transition_params, [1, 0, 2, 3])
        self._num_tags = transition_params.get_shape()[2].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None, position=None):
        """Build the CrfDecodeForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      scope: Unused variable scope of this cell.
    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        state = tf.expand_dims(state, 2)  # [B, O, 1]
        if position is None:
            position = tf.constant(0, dtype=tf.int32)
        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension.
        # [B, O, 1] + [1, O, O] -> [B, O, O]
        transition_scores = state + self._transition_params[position]  # [B, O, O]
        new_state = inputs + tf.reduce_max(transition_scores, [1])  # [B, O]
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)  # [B, O]
        return backpointers, new_state


class BigramCrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
       Samw with CrfDecodeBackwardRnnCell
    """

    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCell.
    Args:
      num_tags: An integer. The number of tags.
    """
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.
      scope: Unused variable scope of this cell.
    Returns:
      new_tags, new_tags: A pair of [batch_size, num_tags]
        tensors containing the new tag indices.
    """
        state = tf.squeeze(state, axis=[1])  # [B]
        batch_size = tf.shape(inputs)[0]
        b_indices = tf.range(batch_size)  # [B]
        indices = tf.stack([b_indices, state], axis=1)  # [B, 2]
        new_tags = tf.expand_dims(
            tf.gather_nd(inputs, indices),  # [B]
            axis=-1)  # [B, 1]

        return new_tags, new_tags
