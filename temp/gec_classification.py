import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile
from config import Config
import config
import data as data_feeder
import time
import datetime
import sys
from gcnn_model import WordModel

class Classification(object):
    def __init__(self, model_file, num_min_probability, eval_per_steps=10000, save_per_steps=1000):
        self.FLAGS = config.FLAGS
        self._config = Config()
        self._config.get_config(self.FLAGS.vocab_path, self.FLAGS.model_config)
        self.model = model_file
        self.num_word = int(num_min_probability)
        self.num_setps = self._config.num_steps
        # self.TENSOR_PROBABILITIES = "Online/WordModel/probabilities: 0"
        # self.TENSOR_CNN_OUT = "Online/WordModel/cnn_out: 0"
        # self.TENSOR_LM_INPUT = "Online/WordModel/batched_input_word_ids: 0"
        self.vocab_size = self._config.vocab_size_in
        self.batch_size = self._config.batch_size
        self.hidden_size = self._config.word_hidden_size
        self.eval_per_steps = eval_per_steps
        self.save_per_steps = save_per_steps


    def export_graph(self, session, iter):

        variables_to_export = ["Classification/softmax_and_output/predictions"
                               ]


        graph_def = convert_variables_to_constants(session, session.graph_def, variables_to_export)
        config_name = self.FLAGS.model_config
        model_export_path = os.path.join(self.FLAGS.graph_save_path)
        if not os.path.isdir(model_export_path):
            os.makedirs(model_export_path)
        model_export_name = os.path.join(model_export_path,
                                         config_name[config_name.rfind("/") + 1:] + "-iter" + str(
                                             iter) + "-" + '.pb')
        f = open(model_export_name, "wb")
        f.write(graph_def.SerializeToString())
        f.close()
        print("Graph is saved to: ", model_export_name)


    def get_min_probability_words_cnnout(self):
        inputs_one_hot = tf.one_hot(indices=self.lm.target_data, depth=self.vocab_size, axis=-1)  # [BATCH_SIZE, NUM_STEPS, VOCAB_SIZE]
        probs = tf.reduce_max(tf.reshape(self.lm._probabilities, [self.batch_size, self.num_setps, -1]) * inputs_one_hot, axis=-1)  # [BATCH_SIZE, NUM_STEPS]
        #print("probs shape:", probs.shape)
        zero_mask = tf.cast(tf.equal(self.lm.target_data, tf.zeros_like(self.lm.target_data)), tf.float32)

        probs = probs + zero_mask  # 给补零的位置加一（从而结果大于1），防止被取出
        # probs = probs * (1 - zero_mask) + tf.ones_like(probs) * zero_mask # 若要严格等价于源代码可以这样写，但是没必要

        top_k_values, top_k_indices = tf.nn.top_k(-probs, k=self.num_word)  # [BATCH_SIZE, TOP_K], [BATCH_SIZE, TOP_K]

        # 形如 [BATCH_SIZE * TOP_K]，第 i * TOP_K 至 (i+1)* TOP_K 之间的元素全是 i。
        # 即：[0, 0, ..., 0, 1, 1, ..., 1, ..., BATCH_SIZE-1, BATCH_SIZE-1, ..., BATCH_SIZE-1]
        cnn_indices = tf.reshape(tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), multiples=[1, self.num_word]), [-1])
        res = tf.transpose(tf.stack([cnn_indices, tf.reshape(top_k_indices, [-1])]), [1, 0])  # [BATCH_SIZE * TOP_K, 2]

        # 即手动添加上batch的这个维度，然后将batch和topk提取出的index拼一起，即可正常提取。
        cnn_outputs = tf.gather_nd(params=self.lm.cnn_out, indices=res)  # [BATCH_SIZE * TOP_K, HIDDEN_SIZE]
        #print('cnn_outputs shape:', cnn_outputs.shape)
        cnn_outputs = tf.reshape(cnn_outputs, [self.batch_size, self.num_word, -1])
        print('cnn_outputs shape:', cnn_outputs.shape)
        return cnn_outputs

    def main(self):
        if not self.FLAGS.data_path:
            raise ValueError("Must set --data_path to PTB data directory")

        logfile = open(self.FLAGS.model_config + '.log', 'w')
        train_data = data_feeder.read_file(self.FLAGS.data_path, self._config, is_train=True)
        valid_data = data_feeder.read_file(self.FLAGS.data_path, self._config, is_train=False)

        with tf.variable_scope("WordModel"):
            self.lm = WordModel(is_training=False, config=self._config)
        restore_variables = dict()
        for v in tf.trainable_variables():
            print("LM Variables:", v.name)
            restore_variables[v.name] = v
        gpu_config = tf.ConfigProto()
        #gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.45
        self._sess = tf.Session(config=gpu_config)

        sv = tf.train.Saver(restore_variables)
        sv.restore(self._sess, self.model)

        with tf.variable_scope("Classification"):
            self.global_step = tf.train.get_or_create_global_step()

            self.cnn_inputs = self.get_min_probability_words_cnnout()   #[BATCH_SIZE, TOP_K, HIDDEN_SIZE]
            # self.cnn_inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_word, self.hidden_size], name="cnn_outputs_for_classification")
            self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2], name='y')

            with tf.name_scope("maxpooling_layer"):
                h = tf.expand_dims(self.cnn_inputs, axis=-1)
                pooled = tf.nn.max_pool(h, ksize=[1, self.num_word, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="maxpool")
                pooled = tf.squeeze(pooled)
                #[batch, hideen_size]
                print("The shape of pooling outputs:", pooled.shape)

            with tf.variable_scope("softmax_and_output"):
                self.soft_w = tf.get_variable("soft_w", shape=[self.hidden_size, 2], dtype=tf.float32)
                soft_b = tf.get_variable("soft_b", shape=[2], dtype=tf.float32)

                self.logits = tf.nn.xw_plus_b(pooled, self.soft_w, soft_b, name="logits")

                self.predictions = tf.argmax(self.logits, 1, name="predictions")

            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss = tf.reduce_mean(losses)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.arg_max(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            self._train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss, global_step=self.global_step)

        restore_variables_2 = dict()
        for v in tf.trainable_variables():
            print("Store:", v.name)

            restore_variables_2[v.name] = v

        sv_2 = tf.train.Saver(restore_variables_2)

        if not self.FLAGS.model_name.endswith(".ckpt"):
            self.FLAGS.model_name += ".ckpt"

        new_vars = [v for v in tf.global_variables() if v.name.startswith("Classification")]
        for v in new_vars:
            print("Init:", v)
        init = tf.variables_initializer(new_vars, name="Initializer")
        self._sess.run(init)

        check_point_dir = os.path.join(self.FLAGS.save_path)
        ckpt = tf.train.get_checkpoint_state(check_point_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            sv_2.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")


        max_train_epoch = self._config.max_max_epoch
        save_path = os.path.join(self.FLAGS.save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for i in range(max_train_epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
            print("Epoch: %d" % (i + 1), file=logfile)
            ######################    Train    #########################
            for step, (batches_per_epoch, lm_in, lm_out, label) in enumerate(data_feeder.data_iterator(train_data, self._config)):
                if step >= batches_per_epoch:
                    break
                feed_dict = {self.lm.input_data: lm_in,
                             self.lm.target_data: lm_out,
                             self.y: label}

                _, gloabl_step, loss, accurarcy = self._sess.run([self._train_op, self.global_step, self.loss, self.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: epoch {}, step {}, loss {:g}, acc {:g}" .format(time_str, i+1,gloabl_step, loss, accurarcy))
                if gloabl_step % self.eval_per_steps == 0:
                    print("Evaluation:\n")
                    for step, (batches_per_epoch, lm_in, lm_out, label) in enumerate(
                            data_feeder.data_iterator(valid_data, self._config)):
                        if step >= batches_per_epoch:
                            break
                        feed_dict = {self.lm.input_data: lm_in,
                                     self.lm.target_data: lm_out,
                                     self.y: label}

                        loss, dev_accurarcy = self._sess.run([self.loss, self.accuracy], feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: epoch {}, Develop loss {:g}, acc {:g}".format(time_str, i + 1, loss, dev_accurarcy))

                    print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
                    # print("Epoch: %d Valid acc: %.3f" % (i + 1, dev_accurarcy), file=logfile)
                    logfile.flush()
                if gloabl_step % self.save_per_steps == 0:
                    print("Saving Model:\n")
                    if self.FLAGS.save_path:
                        print("Saving model to %s." % self.FLAGS.save_path, file=logfile)
                        model_save_path = os.path.join(save_path, self.FLAGS.model_name)
                        sv_2.save(self._sess, model_save_path)
                        print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Begin exporting graph!")
                        self.export_graph(self._sess, i)
                        print("[" + time.strftime('%Y-%m-%d %H:%M:%S') + "] Finish exporting graph!")
            print(time.strftime('%Y-%m-%d %H:%M:%S'), file=logfile)
            # print("Epoch: %d Train acc: %.3f" % (i + 1, accurarcy), file=logfile)
            logfile.flush()
        logfile.close()

if __name__ == "__main__":
    args = sys.argv
    model_file = args[1]
    num_min_probability = args[2]
    test_file_out = "test_result"
    classifier = Classification(model_file, num_min_probability)
    classifier.main()

    
















