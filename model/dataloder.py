import numpy as np
import torch
from torch.autograd import Variable
from data import const


class DataLoader(object):

    def __init__(self, ncf, lc, ncl, rc, netcing, netced, label, maxlen, device, batch_size=64, shuffle=True):

        self.device = device
        self.size = len(label)
        self.maxlen = maxlen
        self.batch_size = batch_size

        self._ncf = np.array(ncf)
        self._lc = np.array(lc)
        self._ncl = np.array(ncl)
        self._netcing = np.array(netcing)
        self._netced = np.array(netced)
        self._label = np.array(label)
        self._rc = np.array(rc)

        self._step = 0
        self._stop_step = self.size // self.batch_size

        if shuffle:
            self._shuffle()


    def _shuffle(self):

        indices = np.arange(self.size)
        np.random.shuffle(indices)

        self._ncf = self._ncf[indices]
        self._lc = self._lc[indices]
        self._ncl = self._ncl[indices]
        self._rc = self._rc[indices]
        self._netcing = self._netcing[indices]
        self._netced = self._netced[indices]
        self._label = self._label[indices]


    def __iter__(self):

        return self

    def __next__(self):

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self.batch_size
        _bsz = self.batch_size
        self._step += 1

        ncf = self._ncf[_start: _start+_bsz]
        lc = self._lc[_start: _start+_bsz]
        rc = self._rc[_start: _start+_bsz]
        nca = self._ncl[_start: _start+_bsz]
        netcing = self._netcing[_start:_start+_bsz]
        netced = self._netced[_start:_start+_bsz]
        # rc = self._rc[_start:_start+_bsz]
        label = self._label[_start:_start+_bsz]

        return ncf, lc, nca, rc, netcing, netced, label

