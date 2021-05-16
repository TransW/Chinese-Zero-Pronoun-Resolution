import os
import operator
import torch
import torch.nn as nn
import pickle

from data_loader import DataLoader

from model import CZPRModel
from ptm.tokenization_bert import BertTokenizer

from utils import load_model, decode, get_output, get_hit_score
from metric import scorer


class Predictor(object):
    def __init__(self, vocab_path, ptm_vocab_path,
                 model_path, batch_size, use_cuda=False):

        self.vocab = self._load_vocab(vocab_path)
        self.tokenizer = BertTokenizer(ptm_vocab_path)
        self.args, self.model = self._load_model(model_path)
        print(self.args)
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()


    def _load_vocab(self, vocab_path):
        vocab = torch.load(vocab_path)
        return vocab

    def _load_model(self, model_path):
        model, args = load_model(model_path, CZPRModel)
        return model, args
    

    def _get_data_loader(self, data):
        data_loader = DataLoader(self.args, data, self.vocab, self.tokenizer)
        return data_loader


    def interface(self, data):
        data_loader = self._get_data_loader(data)
        all_uuids = []
        all_labels = []
        all_preds = []
        all_probs = []
        for batch in data_loader.batch_iter():
            uuids = batch['uuids']
            label_ids = batch['label_ids']
            logits = self.model(batch)
            label_ids, pred_ids, probs = decode(label_ids, logits)
            all_uuids += uuids
            all_labels += label_ids
            all_preds += pred_ids
            all_probs += probs
        gold, output = get_output(all_uuids, all_labels, all_preds, all_probs)
        return gold, output


def load_data(file_path):
    data = []
    with open(file_path, 'rb') as f:
        examples = pickle.load(f)
    gold = {}
    for example in examples:
        zp_idx = example['zp_idx']
        zp_prefix = example['zp_prefix']
        zp_postfix = example['zp_postfix']
        cands = example['cands']
        gold[zp_idx] = []
        for item in cands:
            cand_idx = item['cand_idx']
            cand = item['cand']
            ifl = item['ifl']
            res = item['res']
            if res == 1:
                gold[zp_idx].append({'res':res, 'cand':cand,
                                     'cand_idx':cand_idx})
            new_example = {'zp_idx':zp_idx, 'zp_pre':zp_prefix, 'zp_post':zp_postfix,
                           'cand_idx':cand_idx, 'cand':cand, 'ifl':ifl, 'label':res}
            data.append(new_example)
    return data, gold




if __name__ == '__main__':

    test_data_path = './dataset/test_examples.pkl'
    vocab_path = './saved/vocab.pt'
    ptm_vocab_path = './saved/bert_base_ch/vocab.txt'
    model_path = './saved/models/test_2/'
    predictor = Predictor(vocab_path, ptm_vocab_path, model_path, batch_size=2)
    test_data, checked = load_data(test_data_path)
    gold, output = predictor.interface(test_data)
    get_hit_score(gold, output)




