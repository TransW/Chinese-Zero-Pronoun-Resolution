# -*- coding: utf-8 -*-

import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn

loss_func = nn.CrossEntropyLoss()


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


def decode(label_ids, logits):
    label_ids = label_ids.tolist()
    softmax = nn.Softmax(dim=-1)
    probs, pred_ids = softmax(logits).max(dim=-1)
    probs = probs.tolist()
    pred_ids = pred_ids.tolist()
    return label_ids, pred_ids, probs


def get_output(uuids, label_ids, pred_ids, probs):
    assert len(uuids) == len(label_ids) == len(pred_ids) == len(probs)
    gold = {}
    output = {}
    for uuid, label_id, pred_id, prob in zip(uuids, label_ids, pred_ids, probs):
        zp_idx, cand_idx = uuid.split('&')
        zp_idx = int(zp_idx)
        cand_idx = int(cand_idx)
        if zp_idx not in gold:
            gold[int(zp_idx)] = []
        if zp_idx not in output:
            output[int(zp_idx)] = []
        if label_id == 1:
            gold[int(zp_idx)].append(uuid)
        if pred_id == 1:
            output[int(zp_idx)].append((uuid, prob))
    return gold, output

def get_hit_score(gold, output):
    gold_count = len(gold)
    hit_count = 0
    pred_count = 0
    for k in output:
        gold_uuids = gold[k]
        output_uuids = output[k]
        output_uuids = sorted(output_uuids, key=lambda x:x[-1], reverse=True)
        if len(output_uuids) != 0:
            output_uuid = output_uuids[0][0]
            pred_count += 1
            if output_uuid in gold_uuids:
                hit_count += 1
    print('Hit Count:{}'.format(hit_count))
    print('Gold Count:{}'.format(gold_count))
    print('Pred Count:{}'.format(pred_count))
    p = hit_count / pred_count
    r = hit_count / gold_count
    f = (2*p*r) / (p+r)
    print('P:{}'.format(p))
    print('R:{}'.format(r))
    print('F:{}'.format(f))
    return f


def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['use_cuda']:
        torch.cuda.manual_seed(args['seed'])


def load_model(model_path, model_class):
    model_saved_path = os.path.join(model_path, 'checkpoint.pt')
    checkpoint = torch.load(model_saved_path, lambda storage, loc: storage)
    args = checkpoint['args']
    model = model_class(args)
    model.load_state_dict(checkpoint['model'], strict=False)
    return args, model


def save_model(args, model, model_path):
    model_saved_path = os.path.join(model_path, 'checkpoint.pt')
    model_state = model.state_dict()
    checkpoint = {
        'model': model_state,
        'args':args
        }
    torch.save(checkpoint, model_saved_path)
    print('Model Saved.')


def is_zp(word):
    return len(word) > 2 and word.count('*') >= 2


def get_long_tensor(batch_ids):
    batch_size = len(batch_ids)
    batch_lens = [len(ids) for ids in batch_ids]
    max_batch_len = max(batch_lens)
    tensor = torch.zeros(batch_size, max_batch_len).long()
    for i, ids in enumerate(batch_ids):
        tensor[i, :len(ids)] = torch.tensor(ids).long()
    return tensor


def get_select_ids(batch_subword_ids):
    batch_size = len(batch_subword_ids)
    batch_true_lens = [len(subword_ids) for subword_ids in batch_subword_ids]
    max_batch_true_len = max(batch_true_lens)
    batch_select_ids = torch.zeros(batch_size, max_batch_true_len).long()
    for i in range(batch_size):
        true_len = batch_true_lens[i]
        subword_ids = batch_subword_ids[i]
        # 取词尾
        select_ids = [item[-1] for item in subword_ids]
        select_ids = torch.tensor(select_ids).long()
        batch_select_ids[i, :true_len] = select_ids
    return batch_select_ids

if __name__ == '__main__':
    batch_ids = [[1,2,3], [4, 5], [2,2,1]]
    tensor = get_long_tensor(batch_ids)

    batch_subword_ids = [[[2, 3], [2]],
                         [[3, 4], [1], [3]],
                         [[2]]]

    tensor = get_select_ids(batch_subword_ids)

