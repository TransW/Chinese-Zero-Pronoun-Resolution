# -*- coding: utf-8 -*-

import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,\
    PackedSequence


class RNNAttn(nn.Module):
    def __init__(self, args):
        super(RNNAttn, self).__init__()

        word_emb_size = args['word_emb_size']
        word_rnn_size = args['word_rnn_size']
        num_word_rnn_layers = args['num_word_rnn_layers']
        word_attn_size = args['word_attn_size']
        dp_rate = args['dp_rate']

        self.word_rnn_layer = nn.GRU(word_emb_size, word_rnn_size,
                                     num_layers=num_word_rnn_layers, bidirectional=True,
                                     dropout=0, batch_first=True)

        self.word_attn_layer = nn.Sequential(nn.Linear(2*word_rnn_size, word_attn_size),
                                             nn.Tanh(),
                                             nn.Linear(word_attn_size, 1, bias=False))

        self.dropout = nn.Dropout(dp_rate)


    def forward(self, word_embs, sent_lens):
        """
        param word_embs: Tensor [num_sents, num_words, word_emb_size]
        param sent_lens: Tensor [num_sents]
        return sent_embs: Tensor [num_sents, 2*word_rnn_size]
        return word_embs: Tensor [num_sents, num_words, 2*word_rnn_size]
        return word_alphas: Tensor [num_sents, num_words]
        """

        sent_lens, sorted_ids = torch.sort(sent_lens, descending=True)
        word_embs = word_embs[sorted_ids]

        word_embs = self.dropout(word_embs)

        # packed_word_embs: [num_true_words, word_emb_size]
        packed_word_embs = pack_padded_sequence(word_embs,
                                                lengths=sent_lens.tolist(),
                                                batch_first=True)

        # packed_word_embs: [num_true_words, 2*word_rnn_size]
        packed_word_embs, _ = self.word_rnn_layer(packed_word_embs)

        # word_attns: [num_true_words]
        word_attns = self.word_attn_layer(packed_word_embs.data).squeeze(1)

        max_value = word_attns.max()
        word_attns = torch.exp(word_attns - max_value)
        word_attns = PackedSequence(data=word_attns,
                                    batch_sizes=packed_word_embs.batch_sizes)

        # word_attns: [num_sents, num_words]
        word_attns, _ = pad_packed_sequence(word_attns, batch_first=True)

        # word_alphas: [num_sents, num_words]
        word_alphas = word_attns / torch.sum(word_attns, dim=1, keepdim=True)

        # word_embs: [num_sents, num_words, 2*word_rnn_size]
        word_embs, _ = pad_packed_sequence(packed_word_embs, batch_first=True)
        word_embs = word_embs * word_alphas.unsqueeze(2)

        # sent_embs: [num_sents, 2*word_rnn_size]
        sent_embs = self.dropout(word_embs.sum(1))

        word_embs = self.dropout(word_embs)

        _, recoverd_ids = torch.sort(sorted_ids, descending=False)
        sent_embs = sent_embs[recoverd_ids]
        word_embs = word_embs[recoverd_ids]
        word_alphas = word_alphas[recoverd_ids]

        return sent_embs, word_embs, word_alphas




class DotAttn(nn.Module):

    def __init__(self, args):
        super(DotAttn, self).__init__()
        key_size = args['key_size']
        dp_rate = args['dp_rate']
        self.key_size = key_size
        self.scale = math.sqrt(key_size)
        self.dropout = nn.Dropout(dp_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        param Q: [batch_size, 1, key_size]
        param K: [batch_size, num_words, key_size]
        param V: [batch_size, num_words, key_size]
        para mask: [batch_size, 1, num_words]
        """

        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if mask is not None:
            output.masked_fill_(mask, -1e12)
        output = self.softmax(output)
        output = torch.matmul(output, V)

        output = self.dropout(output)
        return output







class CZPRModel(nn.Module):
    def __init__(self, args):
        super(CZPRModel, self).__init__()
        word_vocab_size = args['word_vocab_size']

        word_emb_size = args['word_emb_size']
        feature_size = args['feature_size']

        rep_size = args['rep_size']
        dp_rate = args['dp_rate']

        self.word_emb_layer = nn.Embedding(word_vocab_size, word_emb_size,
                                           padding_idx=0)

        self.zp_pre_encoder = RNNAttn(args)
        self.zp_post_encoder = RNNAttn(args)

        self.cand_encoder = RNNAttn(args)

        self.dot_attn_layer = DotAttn(args)

        self.feature_layer = nn.Linear(61, feature_size)

        self.output_layer = nn.Sequential(nn.Linear(feature_size, rep_size),
                                          nn.Tanh(),
                                          nn.Dropout(dp_rate),
                                          nn.Linear(rep_size, 2))

    def load_emb_matrix(self, emb_matrix):
        emb_matrix = nn.Parameter(emb_matrix)
        self.word_emb_layer.weight = emb_matrix


    def forward(self, batch):
        zp_pre_ids = batch['zp_pre_ids']
        zp_pre_lens = batch['zp_pre_lens']

        zp_post_ids = batch['zp_post_ids']
        zp_post_lens = batch['zp_post_lens']

        cand_ids = batch['cand_ids']
        cand_mask = cand_ids.eq(0)
        cand_lens = batch['cand_lens']

        feature_ids = batch['feature_ids']

        zp_pre_word_embs = self.word_emb_layer(zp_pre_ids)
        zp_post_word_embs = self.word_emb_layer(zp_post_ids)
        cand_word_embs = self.word_emb_layer(cand_ids)


        zp_pre_embs, _, _ = self.zp_pre_encoder(zp_pre_word_embs, zp_pre_lens)
        zp_post_embs, _, _ = self.zp_post_encoder(zp_post_word_embs, zp_post_lens)

        cand_embs, cand_word_embs, _ = self.cand_encoder(cand_word_embs, cand_lens)

        cand_zp_pre_embs = self.dot_attn_layer(zp_pre_embs.unsqueeze(1),
                                               cand_word_embs, cand_word_embs,
                                               cand_mask.unsqueeze(1))
        cand_zp_pre_embs = cand_zp_pre_embs.squeeze(1)


        cand_zp_post_embs = self.dot_attn_layer(zp_post_embs.unsqueeze(1),
                                                cand_word_embs, cand_word_embs,
                                                cand_mask.unsqueeze(1))
        cand_zp_post_embs = cand_zp_post_embs.squeeze(1)

        feature_embs = self.feature_layer(feature_ids)

        rep_embs = zp_pre_embs + zp_post_embs + cand_embs + \
            cand_zp_pre_embs + cand_zp_post_embs + feature_embs

        logits = self.output_layer(rep_embs)

        return logits




if __name__ == '__main__':
    query = torch.randn(2, 1, 3)
    key = torch.randn(2, 5, 3)
    value = key
    mask = torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]).bool()
    mask = mask.unsqueeze(1)
    print(mask.size())
    args = {'key_size':3, 'dp_rate':0}
    model = DotAttn(args)
    output = model(query, key, value, mask)
    print(output.size())








