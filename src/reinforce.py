# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Reinforcement learning via Policy Gradient (REINFORCE).
"""

import argparse
import pdb
import random
import re
import time
import logging
import os
import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

import config
import data
import utils
from engine import Engine
from utils import ContextGenerator, ContextGeneratorEval
from agent import LstmAgent, LstmRolloutAgent, RlAgent
from dialog import Dialog, DialogEval, DialogLogger
from record import record

logging.basicConfig(format=config.log_format, level=config.log_level)

class Reinforce(object):
    """Facilitates a dialogue between two agents and constantly updates them."""
    def __init__(self, dialog, ctx_gen, args, engine, corpus, dialog_eval, ctx_gen_eval, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.engine = engine
        self.corpus = corpus
        self.dialog_eval = dialog_eval
        self.ctx_gen_eval = ctx_gen_eval
        self.logger = logger if logger else DialogLogger()

        # record
        self.record_func = record
        if self.args.record_freq > 0:
            if not os.path.exists(self.args.record_path):
                os.mkdir(self.args.record_path)
            self.ppl_exp_file = open(os.path.join(self.args.record_path, 'ppl.log'), 'w')
            self.rl_exp_file = open(os.path.join(self.args.record_path, 'rl.log'), 'w')
            self.text_exp_file = open(os.path.join(self.args.record_path, 'text.json'), 'w')
            self.learning_exp_file = open(os.path.join(self.args.record_path, 'learning.log'), 'w')

    def run(self):
        """Entry point of the training."""
        validset, validset_stats = self.corpus.valid_dataset(self.args.bsz,
            device_id=self.engine.device_id)
        trainset, trainset_stats = self.corpus.train_dataset(self.args.bsz,
            device_id=self.engine.device_id)
        N = len(self.corpus.word_dict)

        n = 0
        for ctxs in self.ctx_gen.iter(self.args.nepoch):
            n += 1
            # supervised update
            if self.args.sv_train_freq > 0 and n % self.args.sv_train_freq == 0:
                self.engine.train_single(N, trainset)

            self.logger.dump('=' * 80)
            # run dialogue, it is responsible for reinforcing the agents
            _, _, rl_reward, rl_stats = self.dialog.run(ctxs, self.logger)
            alice_rew, alice_unique = rl_stats['alice_rew'], rl_stats['alice_unique'] 
            if self.args.record_freq > 0 and n % self.args.record_freq == 0:
                self.learning_exp_file.write('{}\t{}\t{}\n'.format(n, alice_rew, alice_unique))
                self.learning_exp_file.flush()
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)
                logging.info('%d: %s' % (n, self.dialog.show_metrics()))

            if self.args.record_freq > 0 and n % self.args.record_freq == 0:
                print('-'*15, 'Recording start', '-'*15)
                self.record_func(n, self.engine, N, validset, validset_stats, self.ppl_exp_file, \
                                    self.dialog_eval, self.ctx_gen_eval, self.rl_exp_file, self.text_exp_file)
                print('-'*15, 'Recording end', '-'*15)

        def dump_stats(dataset, stats, name):
            loss, select_loss = self.engine.valid_pass(N, dataset, stats)
            self.logger.dump('final: %s_loss %.3f %s_ppl %.3f' % (
                name, float(loss), name, np.exp(float(loss))),
                forced=True)
            self.logger.dump('final: %s_select_loss %.3f %s_select_ppl %.3f' % (
                name, float(select_loss), name, np.exp(float(select_loss))),
                forced=True)

        dump_stats(trainset, trainset_stats, 'train')
        dump_stats(validset, validset_stats, 'valid')

        self.logger.dump('final: %s' % self.dialog.show_metrics(), forced=True)


def main():
    parser = argparse.ArgumentParser(description='Reinforce')
    parser.add_argument('--data', type=str, default=config.data_dir,
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=config.unk_threshold,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--output_model_file', type=str,
        help='output model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=config.temperature,
        help='temperature')
    parser.add_argument('--cuda', action='store_true', default=config.cuda,
        help='use CUDA')
    parser.add_argument('--verbose', action='store_true',
        help='print out converations')
    parser.add_argument('--seed', type=int, default=config.seed,
        help='random seed')
    parser.add_argument('--score_threshold', type=int,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--log_file', type=str, default='',
        help='log successful dialogs to file for training')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--gamma', type=float,
        help='discount factor')
    parser.add_argument('--eps', type=float,
        help='eps greedy')
    parser.add_argument('--nesterov', action='store_true', default=config.nesterov,
        help='enable nesterov momentum')
    parser.add_argument('--momentum', type=float,
        help='momentum for sgd')
    parser.add_argument('--lr', type=float,
        help='learning rate')
    parser.add_argument('--clip', type=float,
        help='gradient clip')
    parser.add_argument('--rl_lr', type=float,
        help='RL learning rate')
    parser.add_argument('--rl_clip', type=float,
        help='RL gradient clip')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--bsz', type=int,
        help='batch size')
    parser.add_argument('--sv_train_freq', type=int,
        help='supervision train frequency')
    parser.add_argument('--nepoch', type=int,
        help='number of epochs')
    parser.add_argument('--visual', action='store_true', default=config.plot_graphs,
        help='plot graphs')
    parser.add_argument('--domain', type=str, default=config.domain,
        help='domain for the dialogue')
    parser.add_argument('--record_freq', type=int, default=50,
        help='record frequency')
    parser.add_argument('--record_path', type=str,
        help='record path')
    parser.add_argument('--selfplay_eval_path', type=str, 
        help='selfplay path for evaluation')
    args = parser.parse_args()

    device_id = utils.use_cuda(args.cuda)
    logging.info("Starting training using pytorch version:%s" % (str(torch.__version__)))
    logging.info("CUDA is %s" % ("enabled. Using device_id:"+str(device_id) + " version:" \
        +str(torch.version.cuda) + " on gpu:" + torch.cuda.get_device_name(0) if args.cuda else "disabled"))

    alice_model = utils.load_model(args.alice_model_file)
    # we don't want to use Dropout during RL
    alice_model.eval()
    # Alice is a RL based agent, meaning that she will be learning while selfplaying
    logging.info("Creating RlAgent from alice_model: %s" % (args.alice_model_file))
    alice = RlAgent(alice_model, args, name='Alice')

    # we keep Bob frozen, i.e. we don't update his parameters
    logging.info("Creating Bob's (--smart_bob) LstmRolloutAgent" if args.smart_bob \
        else "Creating Bob's (not --smart_bob) LstmAgent" )
    bob_ty = LstmRolloutAgent if args.smart_bob else LstmAgent
    bob_model = utils.load_model(args.bob_model_file)
    bob_model.eval()
    bob = bob_ty(bob_model, args, name='Bob')

    logging.info("Initializing communication dialogue between Alice and Bob")
    dialog = Dialog([alice, bob], args)
    logger = DialogLogger(verbose=args.verbose, log_file=args.log_file)
    ctx_gen = ContextGenerator(args.context_file)

    # simulation module
    dialog_eval = DialogEval([alice, bob], args)
    ctx_gen_eval = ContextGeneratorEval(args.selfplay_eval_path)

    logging.info("Building word corpus, requiring minimum word frequency of %d for dictionary" % (args.unk_threshold))
    corpus = data.WordCorpus(args.data, freq_cutoff=args.unk_threshold)
    engine = Engine(alice_model, args, device_id, verbose=False)

    logging.info("Starting Reinforcement Learning")
    reinforce = Reinforce(dialog, ctx_gen, args, engine, corpus, dialog_eval, ctx_gen_eval, logger)
    reinforce.run()

    logging.info("Saving updated Alice model to %s" % (args.output_model_file))
    utils.save_model(alice.model, args.output_model_file)


if __name__ == '__main__':
    main()
