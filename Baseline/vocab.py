# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict

from common.base_vocab import BaseVocab, BaseMultiVocab
from common.base_vocab import VOCAB_PREFIX, EMPTY, EMPTY_ID


class CharVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([char for sent in self.data
                           for word in sent
                           for char in word[self.idx]])
        keys = list(counter.keys())
        keys = list(sorted(keys, key=lambda k: counter[k], reverse=True))
        self._id2unit = VOCAB_PREFIX + keys
        self._unit2id = {u:i for i, u in enumerate(self._id2unit)}


class LabelVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([word[self.idx] for sent in self.data
                           for word in sent])
        keys = list(counter.keys())
        keys = list(sorted(keys, key=lambda k: counter[k], reverse=True))
        self._id2unit = keys
        self._unit2id = {u:i for i, u in enumerate(self._id2unit)}


class WordVocab(BaseVocab):
    def __init__(self, data=None, lang='', idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)
        self.state_attrs += ['ignore']

    def id2unit(self, id):
        if len(self.ignore) > 0 and id == EMPTY_ID:
            return '_'
        else:
            return super().id2unit(id)

    def unit2id(self, unit):
        if len(self.ignore) > 0 and unit in self.ignore:
            return self._unit2id[EMPTY]
        else:
            return super().unit2id(unit)

    def build_vocab(self):
        if self.lower:
            counter = Counter([word[self.idx].lower()
                               for sent in self.data
                               for word in sent])
        else:
            counter = Counter([word[self.idx]
                               for sent in self.data
                               for word in sent])

        # 删除指定Unit
        for k in list(counter.keys()):
            if counter[k] < self.cutoff or k in self.ignore:
                del counter[k]

        keys = list(counter.keys())
        keys = list(sorted(keys, key=lambda k: counter[k], reverse=True))
        self._id2unit = VOCAB_PREFIX + keys
        self._unit2id = {u:i for i, u in enumerate(self._id2unit)}



class MultiVocab(BaseMultiVocab):

    def state_dict(self):
        """
        存储类
        """
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state['_key2class'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """
        加载类
        """
        class_dict = {
            'CharVocab': CharVocab,
            'WordVocab': WordVocab,
        }
        new = cls()
        assert '_key2class' in state_dict, 'Cannot find class name mapping in state dict!'
        key2class = state_dict.pop('_key2class')
        for k, v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new


if __name__ == '__main__':
    pass





