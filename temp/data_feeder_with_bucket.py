import numpy as np
import random
from collections import namedtuple
import os
from data_utility import DataUtility
Data = namedtuple('Data', ['in_data', 'in_data_lemma', 'lemma_index', 'in_data_letter',
                           'words_num', 'letters_num', 'out_data'])


class DataFeederContext(object):
    def __init__(self, config, is_train=True, vocab_path="../lang-8_process/user_data/",
                 data_path="../lang-8_process/user_data/"):
        # Use bucketing to reduce padding
        vocab_file_in_words = os.path.join(vocab_path, "vocab_in_words")
        vocab_file_in_letters = os.path.join(vocab_path, "vocab_in_letters")
        vocab_file_out = os.path.join(vocab_path, "vocab_out")

        phase = "train" if is_train else "dev"

        corpus_file_in_letters = os.path.join(data_path, phase + "_in_ids_letters")
        corpus_file_in_lm = os.path.join(data_path, phase + "_in_ids_lm")

        self.PAD_ID = 0
        self.Buckets = config.buckets
        self.num_steps = config.num_steps

        self.data_utility = DataUtility(vocab_file_in_words=vocab_file_in_words,
                                        vocab_file_in_letters=vocab_file_in_letters,
                                        vocab_file_out=vocab_file_out)
        self.all_data = [[] for _ in self.Buckets]
        corpus_in_words_lm = self.load_corpus(corpus_file_in_lm)
        corpus_in_letters = self.load_corpus(corpus_file_in_letters)

        assert len(corpus_in_words_lm) == len(corpus_in_letters)
        for i in range(len(corpus_in_words_lm)):
            corpus_in_words = corpus_in_words_lm[i].strip().split("#")
            lemma_words = corpus_in_words[0].split()
            words = corpus_in_words[1].split()

            if len(corpus_in_words) == 3:
                lemma_index = corpus_in_words[2].split()
            else:
                lemma_index = [0]

            letters = [letter.split() for letter in corpus_in_letters[i].strip().split("#")]

            self.gen_data(words, lemma_words, lemma_index, letters)

        self.train_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        print ("bucket size = " + str(self.train_bucket_sizes))
        self.num_samples = float(sum(self.train_bucket_sizes))
        self.train_buckets_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples
                                    for i in range(len(self.train_bucket_sizes))]
        print ("bucket_scale = " + str(self.train_buckets_scale))
        print ("samples num = " + str(self.num_samples))
        self.current_batch_index = [0 for i in range(len(self.Buckets))]
        self.tmp_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        self.tmp_bucket_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples
                                 for i in range(len(self.train_bucket_sizes))]

    def load_corpus(self, corpus_file_in):
        corpus_array = []
        with open(corpus_file_in, mode="r") as f:
            for line in f:
                corpus_array.append(line)
        return corpus_array

    def init_bucket_param(self):
        self.tmp_bucket_sizes = [len(self.all_data[b]) for b in range(len(self.Buckets))]
        self.tmp_bucket_scale = [sum(self.train_bucket_sizes[:i + 1]) / self.num_samples
                                 for i in range(len(self.train_bucket_sizes))]
        self.current_batch_index = [0 for i in range(len(self.Buckets))]
        for i in range(len(self.all_data)):
            random.shuffle(self.all_data[i])

    def gen_data(self, words, lemma_words, lemma_index, letters):
        for bucket_id, bucket_length in enumerate(self.Buckets):
            if len(words) <= bucket_length:
                in_data = words[:self.Buckets[-1]] + [self.PAD_ID] * (bucket_length - len(words))
                out_data = words[1:-1][:self.Buckets[-1]] + [self.PAD_ID] * (bucket_length - len(words[1:-1]))
                lemma_data = lemma_words[:self.Buckets[-1]] + [self.PAD_ID] * (bucket_length - len(lemma_words))
                letter_data = [letter[:self.num_steps] + [self.PAD_ID] * (self.num_steps - len(letter))
                                for letter in letters]
                letters_num = [len(letter[:self.num_steps]) for letter in letters[:self.Buckets[-1]]] + \
                              [self.PAD_ID] * (bucket_length - len(letters))

                while True:
                    if len(letter_data) >= bucket_length:
                        break
                    letter_data.append([self.PAD_ID] * self.num_steps)

                letter_data = letter_data[: bucket_length]
                words_num = len(words[:self.Buckets[-1]])-2

                data = Data(in_data=in_data, in_data_letter=letter_data, in_data_lemma=lemma_data, lemma_index=lemma_index,
                            words_num=words_num, letters_num=letters_num, out_data=out_data)
                self.all_data[bucket_id].append(data)
                break

    def random_choose_bucket(self, batch_size):
        while True:
            tmp_num_samples = float(sum(self.tmp_bucket_sizes))
            self.tmp_bucket_scale = [sum(self.tmp_bucket_sizes[:i + 1]) / tmp_num_samples for i in
                                     range(len(self.train_bucket_sizes))]
            random_number_01 = np.random.random_sample()
            bucket_id = min(
                [i for i in range(len(self.tmp_bucket_scale)) if self.tmp_bucket_scale[i] > random_number_01])
            if self.current_batch_index[bucket_id] + batch_size > self.train_bucket_sizes[bucket_id]:
                self.tmp_bucket_sizes[bucket_id] = 0

            else:
                self.tmp_bucket_sizes[bucket_id] -= batch_size
                return bucket_id

    def maskWeight(self, letter_num, letter_data, out_data):
        if letter_num == 0:
            return False
        in_letters_id = letter_data[:letter_num]
        in_letters = [self.data_utility.id2token_in_letters[int(id)] for id in in_letters_id]
        in_word = ''.join(in_letters)
        out_word = self.data_utility.id2token_out[int(out_data)]
        if in_word != out_word:
            return 1.0
        else:
            return 2.0
#        mask_num_steps = int(self.data_utility.out_words_count / 3)
#        if int(out_data) <= mask_num_steps:
#            return 1
#        elif mask_num_steps < int(out_data) <= 2 * mask_num_steps:
#            return 1
#        else:
#            return 1

    def lemma_mask(self, index, lemma_data, out_data):

        if int(lemma_data[index]) == int(out_data[index]):
            return 1
        else:
            return 1

    def next_batch_fixmask(self, batch_size=1):
        bucket_id = self.random_choose_bucket(batch_size)
        current_batch_length = self.Buckets[bucket_id]
        start_index, end_index = self.current_batch_index[bucket_id], self.current_batch_index[bucket_id] + batch_size
        data_batch = self.all_data[bucket_id][start_index:end_index]
        input_array = np.array([data.in_data for data in data_batch], dtype=np.int32)
        lemma_input_array = np.array([data.in_data_lemma for data in data_batch], dtype=np.int32)
        seq_len = [data.words_num for data in data_batch]
        output_array = np.array([data.out_data for data in data_batch], dtype=np.int32)
        mask_array = [[1.0] * data.words_num + [0.0] * (current_batch_length - data.words_num)
                      for data in data_batch]
        lemma_mask_array = [[0.0] * current_batch_length for _ in data_batch]
        for i in range(len(lemma_mask_array)):
            for index in data_batch[i].lemma_index:
                lemma_mask_array[i][int(index)] = \
                    self.lemma_mask(int(index), data_batch[i].in_data_lemma, data_batch[i].out_data)
        lemma_mask_array = np.array(lemma_mask_array).reshape([-1])
        input_array_letter = np.array([data.in_data_letter for data in data_batch],
                                      dtype=np.int32).reshape(batch_size * current_batch_length, self.num_steps)

        output_array_letter = np.array([data.out_data for data in data_batch]).reshape([-1])

        mask_array_letter = np.array([[(self.maskWeight(letter_num, letter_data, out_data) if letter_num > 0 else 0.0)
                                       for (letter_num, letter_data, out_data) in
                                       zip(data.letters_num, data.in_data_letter, data.out_data)] for data in data_batch],
                                     dtype=np.float32).reshape([-1])

        seq_len_letter = np.array([data.letters_num for data in data_batch], dtype=np.int32).reshape(batch_size * current_batch_length)

        self.current_batch_index[bucket_id] += batch_size
        word_data = [lemma_input_array, input_array, output_array, mask_array, lemma_mask_array, seq_len]
        letter_data = [input_array_letter, output_array_letter, mask_array_letter, seq_len_letter]
        return word_data, letter_data, current_batch_length


if __name__=="__main__":
    DataReader = DataFeederContext(vocab_path="../lang-8_process/lang-8_data/", data_path="../lang-8_process/lang-8_data/")
    DataReader.next_batch_fixmask()
