# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import os

import sys

from seq2word_split_model_att_v2_relu import WordModel, LetterModel
from config import Config
import config
from data_feeder_with_bucket import DataFeederContext


FLAGS = config.FLAGS
data_type = config.data_type
index_data_type = config.index_data_type
np_index_data_type = config.np_index_data_type


def run_letter_epoch(session, word_models, letter_model, config, eval_op=None, data_feeder=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0
    buckets = config.buckets
    batch_size = config.batch_size
    fetches = {}
    fetches_letter = {}
    epoch_size = int(data_feeder.num_samples // batch_size) - len(buckets)
    data_feeder.init_bucket_param()
    for step in range(epoch_size):
        if step >= epoch_size // FLAGS.laptop_discount:
            break
        word_data, letter_data, batch_length = data_feeder.next_batch_fixmask(batch_size)
        word_model = word_models[buckets.index(batch_length)]

        fetches["output"] = word_model.final_output

        fetches_letter["cost"] = letter_model.loss

        if eval_op is not None:
            fetches_letter["eval_op"] = eval_op
        feed_dict = {word_model.input_data: word_data[1],
                     word_model.target_data: word_data[2],
                     word_model.output_masks: word_data[3],
                     word_model.sequence_length: word_data[5]}
        vals = session.run(fetches, feed_dict)
        feed_dict_letter = {letter_model.word_output: vals["output"],
                            letter_model.input_x: letter_data[0],
                            letter_model.input_y: letter_data[1],
                            letter_model.mask: letter_data[2]
                           }
        vals_letter = session.run(fetches_letter, feed_dict_letter)
        cost = vals_letter["cost"]

        costs += cost
        iters += np.sum(letter_data[2])
        num_word += np.sum(letter_data[3])

        if verbose and step % (epoch_size // 100) == 0:

            print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] %.3f ppl: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters), num_word / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)


def run_word_epoch(session, word_models, config, eval_op=None, data_feeder=None, phase="lm", verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_word = 0
    buckets = config.buckets
    batch_size = config.batch_size
    fetches = {}
    epoch_size = int(data_feeder.num_samples // batch_size) - len(buckets)
    data_feeder.init_bucket_param()

    for step in range(epoch_size):
        if step >= epoch_size // FLAGS.laptop_discount:  # Early stopping for debug purpose
            break
        word_data, _, batch_length = data_feeder.next_batch_fixmask(batch_size)
        # print(batch_length)
        word_model = word_models[buckets.index(batch_length)]
        if phase=="lm":
            fetches["cost"] = word_model.cost[0]
        else:
            fetches["cost"] = word_model.cost[1]

        if eval_op is not None:
            if phase=="lm":
                fetches["eval_op"] = eval_op[buckets.index(batch_length)][0]
            else:
                fetches["eval_op"] = eval_op[buckets.index(batch_length)][1]

        feed_dict = {word_model.lemma_input_data: word_data[0],
                     word_model.input_data: word_data[1],
                     word_model.target_data: word_data[2],
                     word_model.output_masks: word_data[3],
                     word_model.lemma_masks: word_data[4],
                     word_model.sequence_length: word_data[5]
                     }
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        if phase == "lm":
            iters += np.sum(word_data[3])
        else:
            iters += np.sum(word_data[4])

        num_word += np.sum(word_data[5])
        if verbose and step % (epoch_size // 100) == 0:
            print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] "
                  "%.3f word perplexity: %.3f speed: %.0f wps"
                  % (step * 1.0 / epoch_size, np.exp(costs / iters),
                  num_word / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    
    logfile = open(FLAGS.model_config + '.log', 'w')

    config = Config()
    config.get_config(FLAGS.vocab_path, FLAGS.model_config)

    test_config = Config()
    test_config.get_config(FLAGS.vocab_path, FLAGS.model_config)
    test_config.batch_size = 1
    test_config.num_steps = 15
    buckets = config.buckets

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
        with tf.Session(config=gpu_config) as session:
            with tf.name_scope("Train"):
                train_feeder = DataFeederContext(config, is_train=True,
                                                 vocab_path=FLAGS.vocab_path, data_path=FLAGS.data_path)
                mtrain = []
                train_op_array = []
                for j, bucket in enumerate(buckets):
                    with tf.variable_scope("WordModel", reuse=True if j > 0 else None, initializer=initializer):
                        mtrain.append(WordModel(is_training=True, config=config, bucket=bucket))
                        train_op_array.append(mtrain[j].train_op)

                with tf.variable_scope("LetterModel", reuse=None, initializer=initializer):
                    mtrain_letter = LetterModel(is_training=True, config=config)
                    train_letter_op = mtrain_letter.train_op

            with tf.name_scope("Valid"):
                valid_feeder = DataFeederContext(config, is_train=False,
                                                    vocab_path=FLAGS.vocab_path, data_path=FLAGS.data_path)
                mvalid = []
                for j, bucket in enumerate(buckets):
                    with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                        mvalid.append(WordModel(is_training=False, config=config, bucket=bucket))

                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    mvalid_letter = LetterModel(is_training=False, config=config)

            with tf.name_scope("Online"):
                
                with tf.variable_scope("WordModel", reuse=True, initializer=initializer):
                    monline = WordModel(is_training=False, config=test_config, bucket=30)
                with tf.variable_scope("LetterModel", reuse=True, initializer=initializer):
                    monline_letter = LetterModel(is_training=False, config=test_config)

            restore_variables = dict()
            restore_variables2 = dict()
            for v in tf.trainable_variables():

                print("store:", v.name)
                restore_variables[v.name] = v
                if v.name.startswith("WordModel/Lm/BiRNN/fw/Attention_layer") or \
                    v.name.startswith("WordModel/Lm/BiRNN/bw/Attention_layer") or \
                    v.name.startswith("WordModel/LemmaSoftmax"):
                    continue
                restore_variables2[v.name] = v
            sv = tf.train.Saver(restore_variables)
            sv2 = tf.train.Saver(restore_variables2)
            if not FLAGS.model_name.endswith(".ckpt"):
                FLAGS.model_name += ".ckpt"

            session.run(tf.global_variables_initializer())
            
            check_point_dir = os.path.join(FLAGS.save_path)
            ckpt = tf.train.get_checkpoint_state(check_point_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                sv2.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
            save_path = os.path.join(FLAGS.save_path)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            if FLAGS.phase == "lm":
                print("training language model.")
                print("training language model", file=logfile)

                for i in range(config.max_max_epoch // FLAGS.laptop_discount):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    for j in range(len(buckets)):
                        mtrain[j].assign_lr(session, config.learning_rate * lr_decay)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain[0].lr)), file=logfile)
                    train_perplexity = run_word_epoch(session, mtrain, config, train_op_array,
                                                 data_feeder=train_feeder, phase="lm", verbose=True)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    valid_perplexity = run_word_epoch(session, mvalid, config, phase="lm", data_feeder=valid_feeder)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    if FLAGS.save_path:
                        print("Saving model to %s." % FLAGS.save_path, file=logfile)
                        step = mtrain[0].get_global_step(session)
                        model_save_path = os.path.join(save_path, FLAGS.model_name)
                        sv.save(session, model_save_path, global_step=step)

            if FLAGS.phase == "letter":
                print("training letter model.")
                print("training letter model", file=logfile)
                for i in range(config.max_max_epoch // FLAGS.laptop_discount):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                   
                    mtrain_letter.assign_lr(session, config.learning_rate * lr_decay)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    train_ppl = run_letter_epoch(session, mtrain, mtrain_letter,  config, train_letter_op,
                                                     data_feeder=train_feeder,
                                                     verbose=True)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train ppl: %.3f" % (i + 1, train_ppl), file=logfile)
                    logfile.flush()

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    valid_ppl = run_letter_epoch(session, mvalid, mvalid_letter, config, data_feeder=valid_feeder)
                    print("Epoch: %d Valid ppl: %.3f" % (i + 1, valid_ppl), file=logfile)
                    logfile.flush()

                    print("Saving model to %s." % FLAGS.save_path, file=logfile)
                    step = mtrain_letter.get_global_step(session)
                    model_save_path = os.path.join(save_path, FLAGS.model_name)
                    sv.save(session, model_save_path, global_step=step)

            if FLAGS.phase == "softmax":
                print("training lemma model.")
                print("training lemma model", file=logfile)
                for i in range(config.max_max_epoch // FLAGS.laptop_discount):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                    for j in range(len(buckets)):
                        mtrain[j].assign_lr(session, config.learning_rate * lr_decay)

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain[0].lr)), file=logfile)
                    train_perplexity = run_word_epoch(session, mtrain, config, train_op_array,
                                                 data_feeder=train_feeder, phase="lemma", verbose=True)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), file=logfile)
                    logfile.flush()

                    valid_perplexity = run_word_epoch(session, mvalid, config, phase="lemma", data_feeder=valid_feeder)
                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), file=logfile)
                    logfile.flush()

                    print("Saving model to %s." % FLAGS.save_path, file=logfile)
                    step = mtrain[0].get_global_step(session)
                    model_save_path = os.path.join(save_path, FLAGS.model_name)
                    sv.save(session, model_save_path, global_step=step)
            
            logfile.close()


if __name__ == "__main__":
    tf.app.run()
