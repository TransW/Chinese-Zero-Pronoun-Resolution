# -*- coding: utf-8 -*-

import pickle
from buildTree import get_info_from_file
from get_feature import get_fl


def get_words(wl):
    words = []
    for w in wl:
        word = w.word
        words.append(word)
    return words


def load_wd(file_path):
    wd = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            k, v = line.split('\t')
            wd[k] = int(v)
    return wd

def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)



def preprocess(file_path, wd, mode='train'):
    """
    param file_path: 存储训练/测试文档地址的文件
    param wd: 特征字典
    param mode: 训练/测试模式
    """
    paths = [line.strip()
             for line in open(file_path, encoding='utf-8').readlines()]

    total_sentence_num = 0
    all_words = []
    zps_info = []

    is_test = True if 'test' in mode else False


    for path in paths:
        file_name = path.strip()
        if file_name.endswith('onf'):
            print('Processing', file_name)

            zps, azps, cands, nodes_info = get_info_from_file(file_name)

            anaphorics = []
            ana_zps = []

            for (zp_sent_idx, zp_begin_idx, zp_end_idx, antecedents, coref_id, is_real) in azps:
                for (cand_sent_idx, cand_begin_idx, cand_end_idx, coref_id) in antecedents:
                    item_1 = (zp_sent_idx, zp_begin_idx, zp_end_idx,
                            cand_sent_idx, cand_begin_idx, cand_end_idx)
                    anaphorics.append(item_1)
                    item_2 = (zp_sent_idx, zp_begin_idx, zp_end_idx, is_real)
                    ana_zps.append(item_2)


            si2reali = {}
            for k in nodes_info:
                nl, wl = nodes_info[k]
                words = get_words(wl)
                all_words.append(words)
                si2reali[k] = total_sentence_num
                total_sentence_num += 1

            for (zp_sent_idx, zp_begin_idx, zp_end_idx, antecedents, coref_id, is_real) in azps:
                real_zp_sent_idx = si2reali[zp_sent_idx]
                zp = (real_zp_sent_idx, zp_sent_idx, zp_begin_idx, zp_end_idx)
                zp_nl, zp_wl = nodes_info[zp_sent_idx]


                if (zp_sent_idx, zp_begin_idx, zp_end_idx, is_real) not in ana_zps:
                    continue

                if is_test and is_real == 0:
                    continue

                cands_info = []

                for cand_sent_idx in range(max(0, zp_sent_idx - 2), zp_sent_idx + 1):
                    cand_nl, cand_wl = nodes_info[cand_sent_idx]
                    for (cand_begin_idx, cand_end_idx) in cands[cand_sent_idx]:
                        if cand_sent_idx == zp_sent_idx and cand_end_idx > zp_begin_idx:
                            continue

                        res = 0
                        if (zp_sent_idx, zp_begin_idx, zp_end_idx, cand_sent_idx, cand_begin_idx,
                            cand_end_idx) in anaphorics:
                            res = 1
                        real_cand_sent_idx = si2reali[cand_sent_idx]

                        ifl = get_fl((zp_sent_idx, zp_begin_idx, zp_end_idx),
                                     (cand_sent_idx, cand_begin_idx, cand_end_idx),
                                     zp_wl, cand_wl, wd)

                        cand = (real_cand_sent_idx, cand_sent_idx, cand_begin_idx,
                                cand_end_idx, res, -res, ifl)

                        cands_info.append(cand)

                zps_info.append((zp, cands_info))

    return zps_info, all_words


def get_examples(zps_info, all_words):
    examples = []
    zp_idx = 0
    for zp, cands_info in zps_info:
        real_zp_sent_idx, zp_sent_idx, zp_begin_idx, zp_end_idx = zp
        zp_words = all_words[real_zp_sent_idx]

        max_len = len(zp_words)
        zp_prefix = zp_words[max(0, zp_begin_idx - 10): zp_begin_idx]
        zp_postfix = zp_words[zp_end_idx + 1: min(zp_end_idx + 11, max_len)]

        cands = []
        cand_idx = 0
        for real_cand_sent_idx, cand_sent_idx, \
            cand_begin_idx, cand_end_idx, res, target, ifl in cands_info:
                cand_words = all_words[real_cand_sent_idx]
                max_len = len(cand_words)
                cand_prefix = cand_words[max(0, cand_begin_idx - 10): cand_begin_idx]
                cand_postfix = cand_words[cand_end_idx + 1: min(cand_end_idx + 11, max_len)]
                cand = cand_words[cand_begin_idx: cand_end_idx + 1]
                if len(cand) >= 8:
                    cand = cand[-8:]
                item = {'cand_idx':cand_idx, 'cand':cand, 'cand_prefix':cand_prefix,
                        'cand_postfix':cand_postfix, 'res':res, 'ifl':ifl}
                cands.append(item)
                cand_idx += 1
        example = {'zp_idx':zp_idx, 'zp_prefix':zp_prefix,
                   'zp_postfix':zp_postfix, 'cands':cands}
        examples.append(example)
        zp_idx += 1

    print(len(examples))
    return examples




if __name__ == '__main__':

    train_data_path = './data/train_list'

    wd = load_wd(file_path='./resource/feature.txt')

    zps_info, all_words = preprocess(train_data_path, wd, mode='train')

    examples = get_examples(zps_info, all_words)

    save_pkl(examples, './train_examples.pkl')


