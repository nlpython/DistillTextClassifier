from transformers import BertTokenizer
from utils.hyperParams import get_parser
from utils.tools import InputFeature, LstmFeatures
import jieba


class Processer(object):

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        self.max_seq_length = args.max_seq_length
        self.train_file = args.data_dir + '/train.txt'
        self.test_file = args.data_dir + '/test.txt'
        self.dev_file = args.data_dir + '/dev.txt'
        self.all_file = args.data_dir + '/hotel.txt'

        self.word2idx = None
        self.idx2word = None

    def tokenize(self, text):
        """
        Tokenize text.
        """
        if self.tokenizer is None or not isinstance(self.tokenizer, BertTokenizer):
            raise ValueError('tokenizer is None or is not BertTokenizer')
        return self.tokenizer.convert_tokens_to_ids(text)

    def detokenize(self, ids):
        """
        Detokenize ids.
        """
        if self.tokenizer is None or not isinstance(self.tokenizer, BertTokenizer):
            raise ValueError('tokenizer is None or is not BertTokenizer')
        return self.tokenizer.convert_ids_to_tokens(ids)


    def get_map(self):
        """
        Get the map of word to index and index to word
        """
        self.word2idx = {'[PAD]': 0, '[UKN]': 1}
        self.idx2word = {0: '[PAD]', 1: '[UKN]'}
        with open(self.all_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                text = line.split('\t')[1]
                for word in jieba.cut(text, cut_all=True):
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                        self.idx2word[len(self.idx2word)] = word
        return self.word2idx, self.idx2word


    def convert_examples_to_features(self, mode='train'):
        """
        Convert examples to features.
        """
        if mode == 'train':
            file = self.train_file
        elif mode == 'test':
            file = self.test_file
        elif mode == 'dev':
            file = self.dev_file
        else:
            raise ValueError('mode must be train or test')

        features = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                label = int(line.split('\t')[0])
                tokens = list(line.split('\t')[1])
                tokens = ['[CLS]'] + tokens[:self.max_seq_length - 2] + ['[SEP]']

                seq_len = len(tokens)
                tokens = tokens + ['[PAD]'] * (self.max_seq_length - seq_len)
                token_ids = self.tokenize(tokens)
                mask_ids = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
                type_ids = [0] * self.max_seq_length

                assert len(token_ids) == self.max_seq_length
                assert len(mask_ids) == self.max_seq_length
                assert len(type_ids) == self.max_seq_length

                features.append(
                    InputFeature(
                        input_ids=token_ids,
                        mask_ids=mask_ids,
                        type_ids=type_ids,
                        label=label,
                        text=tokens
                    )
                )

        return features

    def convert_file_to_lstm(self, mode='train'):
        """
        Convert file to data.
        """
        if mode == 'train':
            file = self.train_file
        elif mode == 'test':
            file = self.test_file
        elif mode == 'dev':
            file = self.dev_file
        else:
            raise ValueError('mode must be train or test')

        word2idx, idx2word = self.get_map()
        features = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tokens = line.split('\t')[1]
                label = int(line.split('\t')[0])

                token_idx = [word2idx.get(word, 1) for word in jieba.cut(tokens)]

                token_idx = token_idx[:self.max_seq_length]
                token_idx = token_idx + [0] * (self.max_seq_length - len(token_idx))

                assert len(token_idx) == self.max_seq_length

                features.append(LstmFeatures(
                    input_ids=token_idx,
                    label=label,
                    text=tokens)
                )

        return features





if __name__ == '__main__':
    args = get_parser()
    processer = Processer(args)
    # processer.convert_examples_to_features('train')
    processer.convert_file_to_lstm('train')