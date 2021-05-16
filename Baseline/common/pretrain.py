# -*- coding: utf-8 -*-



import os
import pickle
import torch
import numpy as np
from .base_vocab import BaseVocab, VOCAB_PREFIX




class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}



class Pretrain(object):
    def __init__(self, file_path, vec_file_path=None, max_vocab=-1):
        self.file_path = file_path
        self._vec_file_path = vec_file_path
        self._max_vocab = max_vocab

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab, self._emb = self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self._vocab, self._emb = self.load()
        return self._emb

    def load(self):
        if os.path.exists(self.file_path):
            data = torch.load(self.file_path, lambda storage, loc: storage)
            return data['vocab'], data['emb']
        else:
            return self.read_and_save()

    def read_from_file(self, vec_file_path):
        with open(vec_file_path, 'rb') as f:
            vocab, ori_emb = pickle.load(f)
            rows, cols = ori_emb.shape
            emb = np.zeros((len(VOCAB_PREFIX)+rows, cols), dtype=np.float32)
            for i in range(rows):
                emb[len(VOCAB_PREFIX) + i] = ori_emb[i]
            return vocab, emb


    def read_and_save(self):
        if self._vec_file_path is None:
            raise Exception('Vector file is not provided.')
        print('Reading pretrained vectors from {}...'.format(self._vec_file_path))
        words, emb = self.read_from_file(self._vec_file_path)
        vocab = PretrainedWordVocab(words, lower=True)

        data = {'vocab': vocab, 'emb': emb}
        torch.save(data, self.file_path)
        print('Saved pretrained vocab and vectors to {}'.format(self.file_path))
        return vocab, emb
