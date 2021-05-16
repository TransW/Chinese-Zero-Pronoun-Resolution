# -*- coding: utf-8 -*-


import pickle
import random
import torch
import torch.nn as nn
from common.base_vocab import PAD_ID, EMPTY

from ptm.tokenization_bert import BertTokenizer

from utils import is_zp, get_long_tensor, get_select_ids, loss_func, decode

from model import CZPRModel
from metric import scorer


class DataLoader(object):
    def __init__(self, args, source, vocab, tokenizer, shuffle=False):

        self.args = args
        self.batch_size = args['batch_size']
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.data = self.prepare_data(source)

        if shuffle:
            random.shuffle(self.data)
        self.num_examples = len(self.data)

    def __len__(self):
        if self.num_examples > self.num_examples // self.batch_size * self.batch_size:
            return self.num_examples // self.batch_size + 1
        else:
            return self.num_examples // self.batch_size

    def prepare_data(self, source):
        data = []
        for example in source:
            zp_pre = example['zp_pre']
            if len(zp_pre) == 0:
                zp_pre = [EMPTY]
            zp_idx = example['zp_idx']
            cand_idx = example['cand_idx']
            uuid = '&'.join([str(zp_idx), str(cand_idx)])
            zp_post = example['zp_post']
            cand = example['cand']
            ifl = example['ifl']
            label = example['label']
            data.append([uuid, zp_pre, zp_post, cand, ifl, label])
        return data

    def encode_subwords(self, batch_words):
        batch_input_ids = []
        batch_subword_ids = []
        for words in batch_words:
            subword_ids = []
            subwords = []
            for word in words:
                if is_zp(word):
                    word = '[MASK]'
                tokens = self.tokenizer.tokenize(word)
                subword_ids.append([])
                if len(tokens) == 0:
                    tokens = [self.tokenizer.unk_token]
                for token in tokens:
                    subword_ids[-1].append(len(subwords))
                    subwords.append(token)
            input_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            batch_input_ids.append(input_ids)
            batch_subword_ids.append(subword_ids)
        batch_input_ids = get_long_tensor(batch_input_ids)
        batch_select_ids = get_select_ids(batch_subword_ids)
        return batch_input_ids, batch_select_ids

    def encode_words(self, batch_words):
        batch_word_ids = []
        batch_lens = []
        for words in batch_words:
            batch_lens.append(len(words))
            word_ids = self.vocab['word'].map(words)
            batch_word_ids.append(word_ids)
        batch_word_ids = get_long_tensor(batch_word_ids)
        batch_lens = torch.tensor(batch_lens).long()
        return batch_word_ids, batch_lens


    def encode_labels(self, batch_labels):
        batch_label_ids = torch.tensor(batch_labels).long()
        return batch_label_ids

    def encode_features(self, batch_features):
        batch_feature_ids = torch.tensor(batch_features).float()
        return batch_feature_ids

    def process_batch(self, batch):
        output = {}
        fields = [field for field in zip(*batch)]
        batch_uuids = list(fields[0])
        batch_zp_pres = list(fields[1])
        batch_zp_posts = list(fields[2])
        batch_cands = list(fields[3])
        batch_features = list(fields[4])
        batch_labels = list(fields[5])

        batch_zp_pre_ids, batch_zp_pre_lens = self.encode_words(batch_zp_pres)
        batch_zp_post_ids, batch_zp_post_lens = self.encode_words(batch_zp_posts)
        batch_cand_ids, batch_cand_lens = self.encode_words(batch_cands)

        batch_zp_pre_input_ids, batch_zp_pre_select_ids = self.encode_subwords(batch_zp_pres)
        batch_zp_post_input_ids, batch_zp_post_select_ids = self.encode_subwords(batch_zp_posts)
        batch_cand_input_ids, batch_cand_select_ids = self.encode_subwords(batch_cands)

        batch_feature_ids = self.encode_features(batch_features)

        batch_label_ids = self.encode_labels(batch_labels)

        output['uuids'] = batch_uuids
        output['zp_pre_ids'] = batch_zp_pre_ids
        output['zp_pre_lens'] = batch_zp_pre_lens
        output['zp_post_ids'] = batch_zp_post_ids
        output['zp_post_lens'] = batch_zp_post_lens
        output['cand_ids'] = batch_cand_ids
        output['cand_lens'] = batch_cand_lens

        output['zp_pre_input_ids'] = batch_zp_pre_input_ids
        output['zp_pre_select_ids'] = batch_zp_pre_select_ids
        output['zp_post_input_ids'] = batch_zp_post_input_ids
        output['zp_post_select_ids'] = batch_zp_post_select_ids
        output['cand_input_ids'] = batch_cand_input_ids
        output['cand_select_ids'] = batch_cand_select_ids

        output['feature_ids'] = batch_feature_ids
        output['label_ids'] = batch_label_ids

        return output

    def batch_iter(self):
        for i in range(self.num_examples // self.batch_size):
            batch = self.data[i*self.batch_size:(i+1)*self.batch_size]
            batch = self.process_batch(batch)
            yield batch

        if self.num_examples > self.num_examples // self.batch_size * self.batch_size:
            batch = self.data[self.num_examples // self.batch_size * self.batch_size:]
            batch = self.process_batch(batch)
            yield batch



def load_data(file_path):
    data = []
    with open(file_path, 'rb') as f:
        examples = pickle.load(f)
    info = {}
    for example in examples:
        zp_idx = example['zp_idx']
        zp_prefix = example['zp_prefix']
        zp_postfix = example['zp_postfix']
        cands = example['cands']
        info[zp_idx] = []
        for item in cands:
            cand_idx = item['cand_idx']
            cand = item['cand']
            ifl = item['ifl']
            res = item['res']
            if res == 1:
                info[zp_idx].append({'label':res, 'cand':cand,
                                     'cand_idx':cand_idx})

            new_example = {'zp_idx':zp_idx, 'zp_pre':zp_prefix, 'zp_post':zp_postfix,
                           'cand_idx':cand_idx, 'cand':cand, 'ifl':ifl, 'label':res}
            data.append(new_example)
    return data, info




if __name__ == '__main__':

    train_data_path = './dataset/train_examples.pkl'
    test_data_path = './dataset/test_examples.pkl'

    vocab_path = './saved/vocab.pt'
    ptm_vocab_path = './saved/bert_base_ch/vocab.txt'
    vocab = torch.load(vocab_path)

    train_data, _ = load_data(train_data_path)
    test_data, test_info = load_data(test_data_path)


    """
    test_data = test_data[:10]

    tokenizer = BertTokenizer(ptm_vocab_path)

    data_args = {'batch_size':5}
    model_args = {'word_vocab_size':len(vocab['word']),
                  'word_emb_size':10,
                  'word_rnn_size':5,
                  'num_word_rnn_layers':1,
                  'word_attn_size':2,
                  'key_size':10,
                  'feature_size':10,
                  'rep_size':20,
                  'dp_rate':0.3}

    model = CZPRModel(model_args)

    data_loader = DataLoader(data_args, test_data, vocab, tokenizer)

    all_uuids = []
    all_labels = []
    all_preds = []

    for batch in data_loader.batch_iter():
        uuids = batch['uuids']
        label_ids = batch['label_ids']
        logits = model(batch)

        loss = loss_func(logits, label_ids)

        label_ids, pred_ids, probs = decode(label_ids, logits)
        all_uuids += uuids
        all_labels += label_ids
        all_preds += pred_ids

    print(all_uuids)
    scorer(all_labels, all_preds)
    """