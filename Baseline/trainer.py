# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:06:02 2021

@author: shzz0522
"""

from time import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from metric import scorer
from utils import save_model, loss_func, decode, get_output, get_hit_score

LOG_STR = '{}: step {}/{}, loss = {} ({} sec/batch), lr: {}'
EVAL_STR = 'step {}: train_loss = {}, dev_score = {}, dev_loss = {}'




class Trainer(object):
    def __init__(self, args, model, optimizer, scheduler):
        self.args = args
        self.use_cuda = args['use_cuda']
        self.model_path = args['model_path']

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_func = loss_func

        self.num_epochs = args['num_epochs']
        self.log_steps = args['log_steps']
        self.eval_steps = args['eval_steps']
        self.max_steps = args['max_steps']
        self.max_steps_before_stop = args['max_steps_before_stop']
        self.gradient_accumulation_steps = args['gradient_accumulation_steps']
        self.max_grad_norm = args['max_grad_norm']
        self.use_early_stop = args['use_early_stop']

        if self.use_cuda:
            self.model.cuda()

        self.global_steps = 0
        self.last_best_steps = 0
        self.train_loss = 0
        self.do_break = False
        self.dev_score_history = []


    def _process_batch(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if self.use_cuda:
                new_batch[k] = v.cuda()
            else:
                new_batch[k] = v
        return new_batch

    def _get_loss(self, batch):
        self.model.train()
        batch = self._process_batch(batch)

        label_ids = batch['label_ids']
        logits = self.model(batch)
        loss = loss_func(logits, label_ids)

        return loss

    def _get_preds(self, batch):
        self.model.eval()
        batch = self._process_batch(batch)
        with torch.no_grad():
            logits = self.model(batch)
        label_ids = batch['label_ids']

        loss_val = loss_func(logits, label_ids).item()
        label_ids, pred_ids, probs = decode(label_ids, logits)

        return label_ids, pred_ids, probs, loss_val

    def _train_model(self, train_data_loader, dev_data_loader):

        for step, batch in enumerate(train_data_loader.batch_iter()):
            start_time = time()
            loss = self._get_loss(batch)
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                self.global_steps += 1
            loss_val = loss.item()
            self.train_loss += loss_val

            # Log
            if self.global_steps % self.log_steps == 0:
                duration = time() - start_time
                current_lr = self.scheduler.get_lr()[0]
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_str = LOG_STR.format(time_str, self.global_steps, self.max_steps,
                                         loss_val, duration, current_lr)
                print('Log:{}'.format(log_str))

            # Eval
            if self.global_steps % self.eval_steps == 0:
                dev_score, dev_loss = self._eval_model(dev_data_loader)
                self.train_loss = self.train_loss / self.eval_steps
                eval_str = EVAL_STR.format(self.global_steps, self.train_loss,
                                           dev_score, dev_loss)
                print('Eval:{}'.format(eval_str))
                self.train_loss = 0
                if len(self.dev_score_history) == 0 or dev_score > max(self.dev_score_history):
                    self.last_best_steps = self.global_steps
                    save_model(self.args, self.model, self.model_path)
                self.dev_score_history += [dev_score]


    def _eval_model(self, dev_data_loader):

        all_uuids = []
        all_labels = []
        all_preds = []
        all_probs = []
        loss_vals = []
        for batch in dev_data_loader.batch_iter():
            uuids = batch['uuids']
            label_ids, pred_ids, probs, loss_val = self._get_preds(batch)
            all_uuids += uuids
            all_labels += label_ids
            all_preds += pred_ids
            all_probs += probs
            loss_vals.append(loss_val)

        print(len(all_labels))
        print(len(all_preds))
        gold, output = get_output(all_uuids, all_labels, all_preds, all_probs)
        score = get_hit_score(gold, output)

        #score = scorer(all_labels, all_preds)
        loss_val = sum(loss_vals) / len(loss_vals)

        return score, loss_val


    def fit(self, train_data_loader, dev_data_loader):
        print('Training...')
        for epoch_idx in range(self.num_epochs):
            self._train_model(train_data_loader, dev_data_loader)

            if self.use_early_stop:
                if self.global_steps - self.last_best_steps >= self.max_steps_before_stop:
                    self.do_break = True
                if self.global_steps >= self.max_steps:
                    self.do_break = True
                if self.do_break:
                    break

        print('Training ended with {} steps.'.format(self.global_steps))
        if self.dev_score_history:
            best_score = max(self.dev_score_history)*100
            print('Best Score:{}'.format(best_score))



if __name__ == '__main__':
    pass
