import codecs
import numpy as np

def read_data(data_path_pos, data_path_neg, max_sequence_length):
    x = list()
    targets = list()
    y = list()
    seq_length = list()
    output_masks = list()
    fluency_masks = list()

    with codecs.open(data_path_pos, 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            words = line.split()
            words = words[:max_sequence_length]
            seqlen = len(words)
            seq_length.append(seqlen - 2)
            words = words + [0] * max(0, max_sequence_length - seqlen)
            target = words[1 : seqlen-1]
            target = target + [0] * (max_sequence_length - len(target))
            x.append(words)
            y.append([1, 0])
            targets.append(target)
            output_mask = [1] * (seqlen - 2) + [0] * (max_sequence_length - (seqlen - 2))
            output_masks.append(output_mask)
            fluency_masks.append(output_mask)

    with codecs.open(data_path_neg, 'r') as f2:
        for line in f2.readlines():
            line = line.strip()
            words = line.split()
            words = words[:max_sequence_length]
            seqlen = len(words)
            seq_length.append(seqlen - 2)
            words = words + [0] * max(0, max_sequence_length - seqlen)
            target = words[1: seqlen - 1]
            target = target + [0] * (max_sequence_length - len(target))
            x.append(words)
            y.append([0, 1])
            targets.append(target)
            output_mask = [1] * (seqlen - 2) + [0] * (max_sequence_length - (seqlen - 2))
            output_masks.append([0] * max_sequence_length)
            fluency_masks.append(output_mask)

    return np.array(x), np.array(targets), np.array(y), np.array(seq_length), np.array(output_masks), np.array(fluency_masks)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]