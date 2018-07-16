from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



class LM_fluency_Conv(object):
    def __init__(self, batch_size, sequence_length, embedding_size, rnn_hidden_size, vocab_size,
                 num_classes, is_training, filter_sizes, num_filters, num_layers=1, learning_rate=0.01,
                 max_grad_norm=5.0, lm_loss_proportion=0.5):

        assert len(filter_sizes) == len(num_filters)

        self.batch_size = batch_size
        self.num_steps = sequence_length
        self.embedding_size = embedding_size
        self.hidden_size = rnn_hidden_size
        self.vocab_size_in = vocab_size
        self.vocab_size_out = vocab_size
        self.num_classes = num_classes
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None],
                                         name="batched_input_word_ids")
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None],
                                          name="batched_output_word_ids")
        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_classes], name="classification_labels")
        self.output_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_output_word_masks")
        self.fluency_masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None],
                                           name="batched_fluency_masks")
        self.sequence_length = tf.placeholder_with_default(input=tf.fill(dims=[self.batch_size], value=self.num_steps),
                                                           shape=[self.batch_size],
                                                           name="batched_input_sequence_length")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        initializer = tf.contrib.layers.variance_scaling_initializer()

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell, output_keep_prob=self.dropout_keep_prob)

        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell for _ in range(num_layers)], state_is_tuple=True)

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [attn_cell for _ in range(num_layers)], state_is_tuple=True)

        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope("Lm"):
            with tf.variable_scope("Embedding"):
                self._embedding = tf.get_variable("embedding", [self.vocab_size_in, self.embedding_size], dtype=tf.float32, initializer=initializer)
                inputs = tf.nn.embedding_lookup(self._embedding, self.input_data)

                embedding_to_rnn = tf.get_variable("embedding_to_rnn",
                                                  [self.embedding_size, self.hidden_size],
                                                  dtype=tf.float32, initializer=initializer)
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]),
                                          embedding_to_rnn),
                                shape=[self.batch_size, -1, self.hidden_size])    #[batch_size, num_steps, hidden_size]

                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)

            with tf.variable_scope("BiRNN"):
                outputs_fw = list()
                state_fw = initial_state_fw
                with tf.variable_scope('fw'):
                    # forward inputs: [<bos>,w1,w2,w3,<eos>,pad,pad...]
                    for timestep in range(self.num_steps):
                        if timestep > 0:
                            tf.get_variable_scope().reuse_variables()
                        (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                        outputs_fw.append(output_fw)

                # backward direction
                outputs_bw = list()
                state_bw = initial_state_bw
                with tf.variable_scope('bw'):
                    inputs = tf.reverse_sequence(inputs, self.sequence_length+2, seq_axis=1, batch_axis=0)
                    # backward inputs: [<eos>,w3,w2,w1,<bos>,pad,pad...]
                    for timestep in range(self.num_steps):
                        if timestep > 0:
                            tf.get_variable_scope().reuse_variables()
                        (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                        outputs_bw.append(output_bw)

                outputs_fw = tf.transpose(outputs_fw, perm=[1, 0, 2])
                outputs_bw = tf.transpose(outputs_bw, perm=[1, 0, 2])
                print("Forward outputs shape:", outputs_fw.shape)
                print("Backward outputs shape:", outputs_bw.shape)

                #截取前seq_length个输出，即舍弃第一个词和bos的输出，并将其反向，以便concat
                outputs_bw = tf.reverse_sequence(outputs_bw, self.sequence_length, seq_axis=1, batch_axis=0)

                output = tf.concat([outputs_fw, outputs_bw], 2)
                output = tf.reshape(output, [-1, self.hidden_size * 2])

                print("output shape:", output.shape)

            with tf.variable_scope("LM_Softmax"):
                rnn_output_to_final_output = tf.get_variable("rnn_output_to_final_output",
                                                                    [self.hidden_size * 2, self.embedding_size],
                                                                    dtype=tf.float32, initializer=initializer)
                self._softmax_w = tf.get_variable("softmax_w", [self.embedding_size, self.vocab_size_out],
                                                  dtype=tf.float32, initializer=initializer)
                softmax_b = tf.get_variable("softmax_b", [self.vocab_size_out], dtype=tf.float32)

        self.final_output = tf.matmul(output, rnn_output_to_final_output)
        logits = tf.matmul(self.final_output, self._softmax_w) + softmax_b
        probabilities = tf.nn.softmax(logits, name="probabilities")

        with tf.variable_scope("Classification"):
            target_oh = tf.one_hot(indices=self.target_data, depth=self.vocab_size_out, axis=-1) #[B, T, V]
            fluency = tf.reduce_max(tf.reshape(logits, [self.batch_size, -1, self.vocab_size_out]) * target_oh, axis=-1)  #[B, T]
            fluency = fluency * self.fluency_masks
            fluency_expanded = tf.expand_dims(fluency, -1) #[B, T, 1]
            fluency_expanded = tf.expand_dims(fluency_expanded, -1) #[B, T, 1, 1]
            print("Fluency_expanded Shape:", fluency_expanded.shape)

            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("Conv-%s" % filter_size):
                    filter_shape = [filter_size, 1, 1, num_filters[i]]
                    W = tf.get_variable("conv_w", shape=filter_shape, dtype=tf.float32, initializer=initializer)
                    b = tf.get_variable("conv_b", shape=[num_filters[i]], dtype=tf.float32)
                    conv = tf.nn.conv2d(fluency_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_relu")
                    pooled = tf.nn.max_pool(h, ksize=[1, self.num_steps-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                            padding="VALID", name="pool")
                    pooled_outputs.append(pooled)
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            print("H_Pool_Flat shape:", self.h_pool_flat.shape)

            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            with tf.variable_scope("Conv_Softmax"):
                conv_soft_w = tf.get_variable("conv_softmax_w", shape=[num_filters_total, num_classes], initializer=initializer)
                conv_soft_b = tf.get_variable("conv_softmax_b", shape=[num_classes])
                self.scores = tf.nn.xw_plus_b(self.h_drop, conv_soft_w, conv_soft_b, name="scores")
                self.classification_predictions = tf.argmax(self.scores, 1, name="classification_predictions")
                correct_predictions = tf.equal(self.classification_predictions, tf.argmax(self.label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        lm_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target_data, [-1])],
                                                                  [tf.reshape(self.output_masks, [-1])],
                                                                  average_across_timesteps=False)
        class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)


        self._cost = cost = (1 - lm_loss_proportion) * tf.reduce_mean(class_loss) + lm_loss_proportion * (tf.reduce_sum(lm_loss) / tf.reduce_sum(self.output_masks))
        self._lm_logits = logits
        self._lm_probabilities = probabilities

        if not is_training:
            return

        self._lr = tf.get_variable(name="learning_rate", shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(learning_rate), trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in tvars:
            print("Variables:", v.name)

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def get_global_step(self, session):
        gs = session.run(self.global_step)
        return gs
