# -*- coding: utf-8 -*-

import json
import pickle
import torch
import numpy as np
import torch.nn as nn

from vocab import CharVocab, WordVocab, LabelVocab, MultiVocab
from common.pretrain import Pretrain
from common.base_vocab import PAD_ID

print(PAD_ID)


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_vocab(data):
    char_vocab = CharVocab(data, idx=0)
    word_vocab = WordVocab(data, idx=0, cutoff=2)
    vocab = MultiVocab({'char': char_vocab,
                        'word': word_vocab})
    return vocab


def build_vocab(file_path):
    all_words = load_pkl(file_path)
    data = []
    for words in all_words:
        words = [[word] for word in words]
        data.append(words)
    vocab = get_vocab(data)
    return vocab



def build_emb_matrix(pretrain, vocab):
    pretrain_vocab = pretrain.vocab
    pretrain_emb = torch.from_numpy(pretrain.emb).float()
    emb_size = pretrain_emb.size()[-1]

    word_vocab = vocab['word']
    word_vocab_size = len(word_vocab)
    pretrain_vocab_size = len(pretrain_vocab)
    common_size = len(set(pretrain_vocab) & set(word_vocab))

    bias = np.sqrt(3.0 / emb_size)
    emb_matrix = torch.zeros(word_vocab_size, emb_size).float()
    nn.init.uniform_(emb_matrix, -bias, bias)

    for word in word_vocab:
        if word in pretrain_vocab:
            word_idx = word_vocab[word]
            pretrain_idx = pretrain_vocab[word]
            emb_matrix[word_idx] = pretrain_emb[pretrain_idx]

    print('Word Vocab size:{}'.format(word_vocab_size))
    print('Pre-trained Vocab size:{}'.format(pretrain_vocab_size))
    print('There are {} words in the Pre-tarined Vocab'.format(common_size))

    return emb_matrix




if __name__ == '__main__':
    # 语料
    all_words_path = './dataset/all_words.pkl'

    # 预训练词向量
    pretrain_path = './saved/pretrain.pt'
    vec_file_path = './resource/wiki_bigram_char.pkl'

    # 词向量矩阵
    emb_matrix_path = './saved/emb_matrix.pt'
    # Vocab
    vocab_path = './saved/vocab.pt'



    # 创建预训练资源
    pretrain = Pretrain(pretrain_path, vec_file_path)


    # 创建 Vocab
    vocab = build_vocab(all_words_path)
    torch.save(vocab, vocab_path)


    # 创建词向量矩阵
    emb_matrix = build_emb_matrix(pretrain, vocab)
    torch.save(emb_matrix, emb_matrix_path)





