import codecs
import sys

def transformer(data_in, data_out, vocab):
    id2tokens = {}
    tokens2id = {}
    with codecs.open(vocab, "r") as f1:
        for line in f1.readlines():
            token, id = line.strip().split("##")
            id = int(id)
            id2tokens[id] = token
            tokens2id[token] = id

    tokens_count = len(tokens2id)

    with codecs.open(data_in, 'r') as f2:
        with codecs.open(data_out, 'w') as f3:
            for line in f2.readlines():
                line = line.strip()
                tokens = line.split()
                f3.write(str(tokens2id["<bos>"]) + ' ')
                for token in tokens:
                    id = tokens2id.get(token, tokens2id["<unk>"])
                    f3.write(str(id) + ' ')
                f3.write(str(tokens2id["<eos>"]) + '\n')

if __name__ == '__main__':
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    vocab = args[3]
    transformer(data_in, data_out, vocab)