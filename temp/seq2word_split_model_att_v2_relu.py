from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import config

FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


class WordModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(self, is_training, config, bucket):
        self.batch_size = config.batch_size
        self.num_steps = bucket
        self.embedding_size = config.word_embedding_size
        self.hidden_size = config.hidden_size
        self.vocab_size_in = config.vocab_size_in
        self.vocab_size_out = config.vocab_size_out
        self.input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.lemma_input_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                               name="batched_lemma_input_word_ids")
        self.target_data = tf.placeholder(dtype=index_data_type(), shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.lemma_masks = tf.placeholder(dtype=tf.float32, shape=[None],
                                           name="batched_lemma_word_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
                                                           shape=[self.batch_size], name="batched_input_sequence_length")
        self.top_k = tf.placeholder(dtype=index_data_type(), shape=[], name="top_k")

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        lstm_state_as_tensor_shape = [config.num_layers, 2, config.batch_size, config.hidden_size]
        
        self._initial_state = tf.placeholder_with_default(tf.zeros(lstm_state_as_tensor_shape, dtype=data_type()),
                                                            lstm_state_as_tensor_shape, name="state")

        unstack_state = tf.unstack(self._initial_state, axis=0)
        tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1])
             for idx in range(config.num_layers)]
        )

        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        self.attention_size = 50

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=data_type())
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)
                lemma_inputs = tf.nn.embedding_lookup(self._embedding, self.lemma_input_data)
                self.lemma_embedding = tf.reshape(lemma_inputs, [-1, self.embedding_size])

                embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                                  [self.embedding_size, self.hidden_size],
                                                  dtype=data_type())
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])

                if is_training and config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, config.keep_prob)

            with tf.variable_scope("BiRNN"):
                outputs_fw = list()
                states_fw = list()
                fw_att_outputs = list()
                state_fw = initial_state_fw
                with tf.variable_scope('fw'):
                    # forward inputs: [<bos>,w1,w2,w3,<eos>,pad,pad...]
                    for timestep in range(self.num_steps):
                        if timestep > 0:
                            tf.get_variable_scope().reuse_variables()
                        (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                        lemma_input = lemma_inputs[:, timestep, :] # (B, D)
                        lemma_input = tf.tile(tf.expand_dims(lemma_input, 1), [1, timestep + 1, 1]) # (B, t, embedding_size)
                        outputs_fw.append(output_fw)
                        f_rnn_output_t = tf.transpose(outputs_fw, perm=[1, 0, 2])  # (B, t, hidden_size)
                        with tf.variable_scope('Attention_layer'):
                            f_w_attention = tf.get_variable('W_Attention', [self.hidden_size, self.embedding_size],
                                                          dtype=data_type())
                            vu = tf.reduce_sum(tf.tensordot(f_rnn_output_t, f_w_attention, 1) * lemma_input, 2) # (B, t)
                            alphas = tf.nn.softmax(vu, name='alphas')  # (B, t)
                            attention_output = tf.reduce_sum(f_rnn_output_t * tf.expand_dims(alphas, -1), 1)  # (B, D)
                            fw_att_outputs.append(tf.concat([attention_output, output_fw], axis=1))
                        states_fw.append(state_fw)
                # backward direction
                outputs_bw = list()
                states_bw = list()
                bw_att_outputs = list()
                state_bw = initial_state_bw

                lemma_inputs = tf.reverse_sequence(lemma_inputs, self.sequence_length, seq_axis=1, batch_axis=0)
                with tf.variable_scope('bw'):
                    inputs = tf.unstack(inputs, axis=0)
                    for batch_index in range(self.batch_size):
                        one_input = inputs[batch_index]
                        seq_len = self.sequence_length[batch_index] + 2
                        inputs[batch_index] = tf.concat(
                            [tf.reverse(one_input[:seq_len, :], [0]), one_input[seq_len:, :]], axis=0)
                    inputs = tf.stack(inputs, axis=0)
                    # backward inputs: [<eos>,w3,w2,w1,<bos>,pad,pad...]
                    for timestep in range(self.num_steps):
                        if timestep > 0:
                            tf.get_variable_scope().reuse_variables()
                        (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                        outputs_bw.append(output_bw)
                        lemma_input = lemma_inputs[:, timestep, :]  # (B, D)
                        lemma_input = tf.tile(tf.expand_dims(lemma_input, 1),
                                              [1, timestep + 1, 1])  # (B, t, embedding_size)
                        b_rnn_output_t = tf.transpose(outputs_bw, perm=[1, 0, 2])  # (B, t, D)
                        with tf.variable_scope('Attention_layer'):
                            b_w_attention = tf.get_variable('W_Attention', [self.hidden_size, self.embedding_size],
                                                           dtype=data_type())
                            vu = tf.reduce_sum(tf.tensordot(b_rnn_output_t, b_w_attention, 1) * lemma_input, 2)  # (B, t)
                            alphas = tf.nn.softmax(vu, name='alphas')  # (B, t)
                            attention_output = tf.reduce_sum(b_rnn_output_t * tf.expand_dims(alphas, -1), 1)  # (B, D)
                            bw_att_outputs.append(tf.concat([attention_output, output_bw], axis=1))
                        states_bw.append(state_bw)

                    # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
                outputs_fw = tf.transpose(outputs_fw, perm=[1, 0, 2])
                outputs_bw = tf.transpose(outputs_bw, perm=[1, 0, 2])

                fw_att_outputs = tf.transpose(fw_att_outputs, perm=[1, 0, 2])
                bw_att_outputs = tf.transpose(bw_att_outputs, perm=[1, 0, 2])

                print("fw att output shape:", fw_att_outputs.shape)
                print("bw att output shape:", bw_att_outputs.shape)

                outputs_bw = tf.unstack(outputs_bw, axis=0)
                for batch_index in range(self.batch_size):
                    one_output = outputs_bw[batch_index]
                    seq_len = self.sequence_length[batch_index]
                    outputs_bw[batch_index] = tf.concat(
                        [tf.reverse(one_output[:seq_len, :], [0]), one_output[seq_len:, :]], axis=0)
                outputs_bw = tf.stack(outputs_bw, axis=0)

                bw_att_outputs = tf.reverse_sequence(bw_att_outputs, self.sequence_length, seq_axis=1, batch_axis=0)

                states_bw = tf.transpose(states_bw, perm=[3, 1, 2, 0, 4])

                states_bw = tf.unstack(states_bw, axis=0)
                for batch_index in range(self.batch_size):
                    one_state = states_bw[batch_index]
                    seq_len = self.sequence_length[batch_index]
                    states_bw[batch_index] = tf.concat(
                        [tf.reverse(one_state[:, :, :seq_len, :], [2]), one_state[:, :, seq_len:, :]], axis=2)
                states_bw = tf.stack(states_bw, axis=0)

                states_fw = tf.transpose(states_fw, perm=[3, 1, 2, 0, 4])
                unstack_states_bw = tf.unstack(states_bw, axis=0)
                unstack_states_fw = tf.unstack(states_fw, axis=0)
                states_fw = tf.concat(unstack_states_fw, axis=2)
                states_bw = tf.concat(unstack_states_bw, axis=2)

                final_state = tf.concat([states_fw, states_bw], 3)
                output = tf.concat([outputs_fw, outputs_bw], 2)
                output = tf.reshape(output, [-1, self.hidden_size * 2])

                att_output = tf.concat([fw_att_outputs, bw_att_outputs], 2)
                att_output = tf.reshape(att_output, [-1, self.hidden_size * 4])

                print("output shape:", output.shape)
                print("state shape:", final_state.shape)

            with tf.variable_scope("Softmax"):
                rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                                    [self.hidden_size * 2, self.embedding_size],
                                                                    dtype=data_type())
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=data_type())
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=data_type())

        with tf.variable_scope("LemmaSoftmax"):
            att_output_to_final_output = tf.get_variable("att_output_to_final_output",
                                                         [self.hidden_size * 4, self.embedding_size],
                                                         dtype=data_type())
            W_matrix = tf.get_variable(
                "W_matrix",
                shape=[self.embedding_size, self.embedding_size],
                dtype=tf.float32)
            U_matrix = tf.get_variable(
                "U_matrix",
                shape=[self.embedding_size, self.embedding_size],
                dtype=tf.float32)

            self.att_final_output = tf.matmul(tf.matmul(att_output, att_output_to_final_output), W_matrix) \
                                    + tf.matmul(self.lemma_embedding, U_matrix)

            lemma_W1 = tf.get_variable(
                "W1",
                shape=[self.embedding_size, self.embedding_size], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            lemma_b1 = tf.get_variable("b1", [self.embedding_size], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))

            lemma_W2 = tf.get_variable(
                "W2",
                shape=[self.embedding_size, self.vocab_size_out], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            lemma_b2 = tf.get_variable("b2", [self.vocab_size_out], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
            att_logits = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(self.att_final_output, lemma_W1, lemma_b1)),
                                         lemma_W2, lemma_b2, name="scores")

        self.final_output = tf.matmul(output, rnn_output_to_final_output)
        logits = tf.matmul(self.final_output, self._softmax_w) + softmax_b
        probabilities = tf.nn.softmax(logits, name="probabilities")
        _, top_k_prediction = tf.nn.top_k(logits, self.top_k, name="top_k_prediction")

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = tf.identity(final_state, "state_out")
        self._logits = logits
        self._probabilities = probabilities
        self._top_k_prediction = top_k_prediction

        att_probabilities = tf.nn.softmax(att_logits, name="att_probabilities")
        _, att_top_k_prediction = tf.nn.top_k(att_logits, self.top_k, name="att_top_k_prediction")

        att_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([att_logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.lemma_masks, [-1])],
                                                                  average_across_timesteps=False)

        self._att_cost = att_cost = tf.reduce_sum(att_loss)
        self._att_logits = att_logits
        self._att_probabilities = att_probabilities
        self._att_top_k_prediction = att_top_k_prediction

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm")

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        att_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="WordModel/Lm/BiRNN/fw/Attention_layer|"
                                                                           "WordModel/Lm/BiRNN/bw/Attention_layer|"
                                                                           "WordModel/LemmaSoftmax")

        att_grads, _ = tf.clip_by_global_norm(tf.gradients(att_cost, att_tvars), config.max_grad_norm)
        att_optimizer = tf.train.GradientDescentOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self._att_train_op = att_optimizer.apply_gradients(zip(att_grads, att_tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def softmax_w(self):
        return self._softmax_w

    @property
    def cost(self):
        return self._cost, self._att_cost

    @property
    def embedding(self):
        return self._embedding

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def logits(self):
        return self._logits, self._att_logits

    @property
    def probalities(self):
        return self._probabilities, self._att_probabilities

    @property
    def top_k_prediction(self):
        return self._top_k_prediction, self._att_top_k_prediction

    @property
    def train_op(self):
        return self._train_op, self._att_train_op


class LetterModel(object):
    """Static PTB model. Modified from old saniti-checked version of dynamic model.
    """

    def __init__(
            self, is_training, config):
        # Placeholders for input, output and dropout
        sequence_length = 20
        num_classes = config.vocab_size_out
        vocab_size = config.vocab_size_letter
        embedding_size = config.letter_embedding_size
        filter_sizes = [3, 4, 5]
        num_filters = 128
        l2_reg_lambda = 0.0

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.word_output = tf.placeholder(tf.float32, [None, config.word_embedding_size], name="word_output")
        self.mask = tf.placeholder(tf.float32, [None], name="mask")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, dtype=tf.float32)

        with tf.variable_scope("CNN"):
            # Embedding layer
            with tf.variable_scope("embedding"):

                self.W = tf.get_variable("W", [vocab_size, embedding_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-1.0, 1.0))

                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            if is_training:
                embedded_chars_expanded = tf.nn.dropout(embedded_chars_expanded, 0.5)
            self.embedded_chars_expanded = embedded_chars_expanded
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % i):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable("W", filter_shape, dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b", [num_filters], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            if is_training:
                self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, 0.5)

            # Final (unnormalized) scores and predictions
            with tf.variable_scope("output"):
                pool_flat_to_embedding_matrix = tf.get_variable(
                    "pool_flat_to_embedding_matrix",
                    shape=[num_filters_total, embedding_size],
                    dtype=tf.float32)
                W = tf.get_variable(
                    "W",
                    shape=[config.letter_embedding_size, num_classes], dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                output_trans_matrix = tf.get_variable("w2n", shape=[config.word_embedding_size + config.letter_embedding_size,
                                                                    config.letter_embedding_size], dtype=tf.float32)
                b = tf.get_variable("b", [num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.output = tf.matmul(self.h_pool_flat, pool_flat_to_embedding_matrix)
                self.output_concated = tf.concat([self.output, self.word_output], axis=1)
               # self.scores = tf.nn.xw_plus_b(self.output_concated, W, b,
               #                               name="scores")
                self.scores = tf.nn.xw_plus_b(tf.matmul(self.output_concated, output_trans_matrix), W, b,
                                                 name="scores")
                # self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
                self.probs = tf.nn.softmax(self.scores, name="probabilities")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                self.predictions = tf.cast(self.predictions, tf.int32)
                self.top_probs = tf.reduce_max(self.probs, 1, name="top_probabilities")

        correct_predictions = tf.equal(self.predictions, self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Calculate mean cross-entropy loss

        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.scores],
                                                                    [self.input_y],
                                                                    [self.mask],
                                                                    average_across_timesteps=False)

        self.loss = tf.reduce_sum(losses) + l2_reg_lambda * l2_loss
        if not is_training:
            return
        
        self.lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LetterModel/CNN")
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})



