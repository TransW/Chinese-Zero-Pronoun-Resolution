# -*- coding: utf-8 -*-


from sklearn.metrics import precision_recall_fscore_support

def scorer(all_labels, all_preds):
    """
    非标准打分函数
    """
    result = precision_recall_fscore_support(all_labels, all_preds,
                                             labels=[0, 1], zero_division=0)
    p = result[0][-1]
    r = result[1][-1]
    f = result[2][-1]
    support = result[3][-1]
    num_examples = len(all_labels)
    detail = {'p':p, 'r':r, 'f':f,
              'support':support, 'num_examples':num_examples}
    print('Detail:{}'.format(detail))
    return f