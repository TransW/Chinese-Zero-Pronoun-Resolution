# -*- coding: utf-8 -*-


import os
import torch

from data_loader import DataLoader
from optim import build_optimizer
from trainer import Trainer

from utils import set_seed, load_data, load_model
from ptm.tokenization_bert import BertTokenizer

from model import CZPRModel
from option import CZPRArgs



def build_model(args):
    if args['trained_model_path']:
        _, model = load_model(args['trained_model_path'], CZPRModel)
    else:
        model = CZPRModel(args)
        if args['use_emb_matrix']:
            emb_matrix = torch.load(args['emb_matrix_path'])
            model.load_emb_matrix(emb_matrix)
    return model


def main(args):
    args = args.to_dict()
    set_seed(args)

    tokenizer = BertTokenizer(args['ptm_vocab_path'])

    print('Load Vocab ...')
    vocab = torch.load(args['vocab_path'])
    args['word_vocab_size'] = len(vocab['word'])

    print('Build DataLoader ...')
    train_data, _ = load_data(args['train_data_path'])
    test_data, _ = load_data(args['test_data_path'])

    if args['debug']:
        train_data = train_data[:]
        test_data = test_data[:]


    train_data_loader = DataLoader(args, train_data,
                                   vocab, tokenizer, shuffle=True)


    test_data_loader = DataLoader(args, test_data,
                                  vocab, tokenizer, shuffle=False)


    print('Build Model & Optimizer ...')
    model = build_model(args)
    optimizer, scheduler = \
        build_optimizer(args, model, train_data_loader)


    trainer = Trainer(args, model, optimizer, scheduler)


    os.makedirs(args['model_path'])
    trainer.fit(train_data_loader, test_data_loader)

if __name__ == '__main__':

    args = CZPRArgs()

    main(args)

