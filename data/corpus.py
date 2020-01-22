import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import numpy as np
from const import *

def word2idx(sents, word2idx):

    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


class Dictionary(object):

    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return '{}(size = {})'.format(self.__class__.__name__, len(self.idx))


class Words(Dictionary):

    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents):
        words = set([word for sent in sents for word in sent])
        # print(words)
        for word in words:
            self._add(word)


class Labels(Dictionary):

    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        _labels = set(labels)
        for label in _labels:
            self._add(label)


class Corpus(object):

    def __init__(self, path, str_maxlen, title_maxlen, cited_maxlen):
        self.train = os.path.join(path, "academic_train")
        self.test = os.path.join(path, "academic_test")
        self.rf_con = os.path.join(path, "academic_papers")
        self._save = os.path.join(path, "academic_corpus")

        self.w = Words()
        self.l = Labels()
        self.str_maxlen = str_maxlen
        self.title_maxlen = title_maxlen
        self.cited_maxlen = cited_maxlen

    @property
    def _paper_dict(self):

        f = open(self.rf_con, 'r')
        paper_dict = {0: '<pad>'}
        for line in f.readlines()[1:]:
            index, title = line.split('\t')[:2]
            paper_dict[int(index)] = title.strip('\n')
        return paper_dict

    def make_dict(self, _file):

        _words, _labels = [], []
        for sentence in open(_file):
            label, _, _, words = sentence.split('\t')
            _words.append(words.split())
            _labels.append(label)

        self.w(_words)
        self.l(_labels)

    def parse_data(self, _file, is_train):

        def seq_integrate(s):
            seqLength = len(s.split())

            if seqLength >= self.str_maxlen:
                return s.split()[:self.str_maxlen]
            else:
                return s.split() + ['<pad>' for _ in range(self.str_maxlen - seqLength)]

        def pad_cited_items(l):

            return l[:self.cited_maxlen] if len(l) >= self.cited_maxlen else l + [PAD] * (self.cited_maxlen-len(l))

        def pad_rc(l):

            pad_rc = []
            for title in l:
                pad_rc += [w for w in title.split() if w in self.w.word2idx.keys()]

            return pad_rc[:self.str_maxlen] if len(pad_rc) >= self.str_maxlen else pad_rc + ['<pad>'] * (self.str_maxlen-len(pad_rc))

        _ncfs, _lcs, _ncls, _rcs, _netcings, _netceds, _labels = [], [], [], [], [], [], []
        for sentence in open(_file):
            label, citing, cited, citation = sentence.split('\t')
            nca, lc, ncf = [seq_integrate(s) for s in citation.split('//')]

            padded_cited = pad_cited_items(eval(cited))
            rc = pad_rc([self._paper_dict[i] for i in padded_cited])

            _ncls.append([self.w.word2idx[i] for i in nca])
            _lcs.append([self.w.word2idx[i] for i in lc])
            _ncfs.append([self.w.word2idx[i] for i in ncf])
            _rcs.append([self.w.word2idx[i] for i in rc])
            _labels.append([self.l.word2idx[label]])
            _netcings.append([int(citing)])
            _netceds.append(padded_cited)

        if is_train:
            self.train_ncf = _ncfs
            self.train_lc = _lcs
            self.train_ncl = _ncls
            self.train_rc = _rcs
            self.train_netcing = _netcings
            self.train_netced = _netceds
            self.train_label = _labels

        else:
            self.test_ncf = _ncfs
            self.test_lc = _lcs
            self.test_ncl = _ncls
            self.test_rc = _rcs
            self.test_netcing = _netcings
            self.test_netced = _netceds
            self.test_label = _labels

    def save(self):

        self.make_dict(self.train)
        self.make_dict(self.test)

        self.parse_data(self.train, is_train=True)
        self.parse_data(self.test, is_train=False)

        data = {
            'maxlen': self.str_maxlen,
            'vocab': self.w.word2idx,
            'label': self.l.word2idx,
            'train': {
                'ncf': self.train_ncf,
                'lc': self.train_lc,
                'ncl': self.train_ncl,
                'rc': self.train_rc,
                'netcing': self.train_netcing,
                'netced': self.train_netced,
                'label': self.train_label,
            },
            'test': {
                'ncf': self.test_ncf,
                'lc': self.test_lc,
                'ncl': self.test_ncl,
                'rc': self.test_rc,
                'netcing': self.test_netcing,
                'netced': self.test_netced,
                'label': self.test_label,
            },
        }

        torch.save(data, self._save)
        print('Finish dumping the data to file - [{}]'.format(self._save))
        print('words length - [{}]'.format(len(self.w)))
        print('label size - [{}]'.format(len(self.l)))


if __name__ == "__main__":

    file_path = os.path.join(BASE_DIR, 'data')
    save_data = os.path.join(BASE_DIR, 'data', 'corpus.pt')
    str_maxlen, title_maxlen, cited_maxlen = 50, 15, 5
    corpus = Corpus(file_path, str_maxlen, title_maxlen, cited_maxlen)
    corpus.save()
