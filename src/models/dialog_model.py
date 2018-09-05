# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
An RNN based dialogue model. Performce both language and choice generation.
"""

import sys
import re
import pdb
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from data import STOP_TOKENS
from domain import get_domain
from models import modules

class DialogModel(modules.CudaModule):
    def __init__(self, word_dict, item_dict, context_dict, output_length, args, device_id):
        super(DialogModel, self).__init__(device_id)

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict
        self.args = args

        # embedding for words
        self.word_encoder = nn.Embedding(len(self.word_dict), args.nembed_word)

        # context encoder
        ctx_encoder_ty = modules.RnnContextEncoder if args.rnn_ctx_encoder \
            else modules.MlpContextEncoder
        self.ctx_encoder = ctx_encoder_ty(len(self.context_dict), domain.input_length(),
            args.nembed_ctx, args.nhid_ctx, args.init_range, device_id)

        # a reader RNN, to encode words
        self.reader = nn.GRU(
            input_size=args.nhid_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)
        self.decoder = nn.Linear(args.nhid_lang, args.nembed_word)
        # a writer, a RNNCell that will be used to generate utterances
        self.writer = nn.GRUCell(
            input_size=args.nhid_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        # tie the weights of reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.dropout = nn.Dropout(args.dropout)

        # a bidirectional selection RNN
        # it will go through input words and generate by the reader hidden states
        # to produce a hidden representation
        self.sel_rnn = nn.GRU(
            input_size=args.nhid_lang + args.nembed_word,
            hidden_size=args.nhid_attn,
            bias=True,
            bidirectional=True)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))

        # attention to combine selection hidden states
        self.attn = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn, args.nhid_attn),
            nn.Tanh(),
            torch.nn.Linear(args.nhid_attn, 1)
        )

        # selection encoder, takes attention output and context hidden and combines them
        self.sel_encoder = nn.Sequential(
            torch.nn.Linear(2 * args.nhid_attn + args.nhid_ctx, args.nhid_sel),
            nn.Tanh()
        )
        # selection decoders, one per each item
        self.sel_decoders = nn.ModuleList()
        for i in range(output_length):
            self.sel_decoders.append(nn.Linear(args.nhid_sel, len(self.item_dict)))

        self.init_weights()

        # fill in the mask
        for i in range(len(self.word_dict)):
            w = self.word_dict.get_word(i)
            special = domain.item_pattern.match(w) or w in ('<unk>', 'YOU:', 'THEM:', '<pad>')
            self.special_token_mask[i] = -999 if special else 0.0

        self.special_token_mask = self.to_device(self.special_token_mask)

    def set_device_id(self, device_id):
        self.device_id = device_id
        self.special_token_mask = self.to_device(self.special_token_mask)

    def zero_hid(self, bsz, nhid=None, copies=None):
        """A helper function to create an zero hidden state."""
        nhid = self.args.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        hid = torch.zeros(copies, bsz, nhid)
        hid = self.to_device(hid)
        return Variable(hid)

    def init_weights(self):
        """Initializes params uniformly."""
        self.decoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)
        self.decoder.bias.data.fill_(0)

        modules.init_rnn(self.reader, self.args.init_range)

        self.word_encoder.weight.data.uniform_(-self.args.init_range, self.args.init_range)

        modules.init_cont(self.attn, self.args.init_range)
        modules.init_cont(self.sel_encoder, self.args.init_range)
        modules.init_cont(self.sel_decoders, self.args.init_range)

    def read(self, inpt, lang_h, ctx_h, prefix_token="THEM:"):
        """Reads a given utterance."""
        # inpt: (max_words, 1)
        # lang_h: (1, 1, 128)
        # ctx_h: (1, 1, nhid_ctx)

        # inpt contains the pronounced utterance
        # add a "THEM:" token to the start of the message
        prefix = Variable(torch.LongTensor(1).unsqueeze(1)) # (1, 1)
        prefix.data.fill_(self.word_dict.get_idx(prefix_token))
        inpt = torch.cat([self.to_device(prefix), inpt]) # (max_words+1, 1)

        # embed words
        inpt_emb = self.word_encoder(inpt) # (max_words+1, 1, nembed_word)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.expand(inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2)) # (max_words+1, 1, nhid_ctx)
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2) # (max_words+1, 1, nembed_word+nhid_ctx)

        # finally read in the words
        self.reader.flatten_parameters()
        out, lang_h = self.reader(inpt_emb, lang_h)
        # out: (max_words+1, 1, nhid_lang)
        # lang_h: (1, 1, 128)

        return out, lang_h

    def word2var(self, word):
        """Creates a variable from a given word."""
        result = Variable(torch.LongTensor(1))
        result.data.fill_(self.word_dict.get_idx(word))
        result = self.to_device(result)
        return result

    def forward_selection(self, inpt, lang_h, ctx_h):
        """Forwards selection pass."""
        # inpt: (max_len-1, <=batch_size)
        # lang_h: (max_len-1, <=batch_size, nhid_lang)
        # ctx_h: (1, <=batch_size, nhid_ctx)
    
        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        inpt_emb = self.word_encoder(inpt) # (max_len-1, <=batch_size, nembed_word)
        h = torch.cat([lang_h, inpt_emb], 2) # (max_len-1, <=batch_size, nhid_lang+nembed_word)
        h = self.dropout(h)

        # runs selection rnn over the hidden state h
        attn_h = self.zero_hid(h.size(1), self.args.nhid_attn, copies=2) # (2, <=batch_size, nhid_attn)

        # compact weights to reduce memory usage
        self.sel_rnn.flatten_parameters()
        h, _ = self.sel_rnn(h, attn_h) # (max_len-1, <=batch_size, 2*nhid_attn)

        # perform attention
        h = h.transpose(0, 1).contiguous() # (<=batch_size, max_len-1, 2*nhid_attn)
        logit = self.attn(h.view(-1, 2 * self.args.nhid_attn)).view(h.size(0), h.size(1)) # (<=batch_size, max_len-1)

        prob = F.softmax(logit,dim=1).unsqueeze(2).expand_as(h) # (<=batch_size, max_len-1, 2*nhid_attn)
        attn = torch.sum(torch.mul(h, prob), 1, keepdim=True).transpose(0, 1).contiguous() # (1, <=batch_size, 2*nhid_attn)

        # concatenate attention and context hidden and pass it to the selection encoder
        h = torch.cat([attn, ctx_h], 2).squeeze(0) # (<=batch_size, 2*nhid_attn+nhid_ctx)
        h = self.dropout(h)
        h = self.sel_encoder.forward(h) # (<=batch_size, nhid_sel)

        # generate logits for each item separately
        outs = [decoder.forward(h) for decoder in self.sel_decoders] # 6*(<=batch_size, len(self.item_dict))
        return torch.cat(outs) # (6*<=batch_size, len(self.item_dict))

    def generate_choice_logits(self, inpt, lang_h, ctx_h):
        """Similar to forward_selection, but is used while selfplaying.
        Thus it is dealing with batches of size 1.
        """
        # inpt: (total_seq_len, 1)
        # lang_h: (total_seq_len, 128 i.e. nhid_lang)
        # ctx_h: (1, 1, nhid_ctx)

        # run a birnn over the concatenation of the input embeddings and
        # language model hidden states
        inpt_emb = self.word_encoder(inpt) # (total_seq_len, 1, nembed_word)
        h = torch.cat([lang_h.unsqueeze(1), inpt_emb], 2) # (total_seq_len, 1, nhid_lang+nembed_word)
        h = self.dropout(h)

        # runs selection rnn over the hidden state h
        attn_h = self.zero_hid(h.size(1), self.args.nhid_attn, copies=2) # (2, 1, nhid_attn)
        self.sel_rnn.flatten_parameters()
        h, _ = self.sel_rnn(h, attn_h) # (total_seq_len, 1, 2*nhid_attn)
        h = h.squeeze(1) # (total_seq_len, 2*nhid_attn)

        # perform attention
        logit = self.attn(h).squeeze(1) # (total_seq_len, )

        prob = F.softmax(logit,dim=0).unsqueeze(1).expand_as(h) # (total_seq_len, 2*nhid_attn)
        attn = torch.sum(torch.mul(h, prob), 0, keepdim=True) # (1, 2*nhid_attn)

        # concatenate attention and context hidden and pass it to the selection encoder
        ctx_h = ctx_h.squeeze(1) # (1, nhid_ctx)
        h = torch.cat([attn, ctx_h], 1) # (1, 2*nhid_attn+nhid_ctx)
        h = self.sel_encoder.forward(h) # (1, nhid_sel)

        # generate logits for each item separately
        logits = [decoder.forward(h).squeeze(0) for decoder in self.sel_decoders] # 6*(len(self.item_dict), )
        return logits

    def write_batch(self, bsz, lang_h, ctx_h, temperature, max_words=100):
        """Generate sentenses for a batch simultaneously."""
        eod = self.word_dict.get_idx('<selection>')

        # resize the language hidden and context hidden states
        lang_h = lang_h.squeeze(0).expand(bsz, lang_h.size(2))
        ctx_h = ctx_h.squeeze(0).expand(bsz, ctx_h.size(2))

        # start the conversation with 'YOU:'
        inpt = torch.LongTensor(bsz).fill_(self.word_dict.get_idx('YOU:'))
        inpt = Variable(self.to_device(inpt))

        outs, lang_hs = [], [lang_h.unsqueeze(0)]
        done = set()
        # generate until max_words are generated, or all the dialogues are done
        for _ in range(max_words):
            # embed the input
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            # pass it through the writer and get new hidden state
            lang_h = self.writer(inpt_emb, lang_h)
            out = self.decoder(lang_h)
            # tie weights with encoder
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # subtract max to make softmax more stable
            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            out = torch.multinomial(scores.exp(), 1).squeeze(1)
            # save outputs and hidden states
            outs.append(out.unsqueeze(0))
            lang_hs.append(lang_h.unsqueeze(0))
            inpt = out

            data = out.data.cpu()
            # check if all the dialogues in the batch are done
            for i in range(bsz):
                if data[i] == eod:
                    done.add(i)
            if len(done) == bsz:
                break

        # run it for the last word to get correct hidden states
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h.unsqueeze(0))

        # concatenate outputs and hidden states into single tensors
        return torch.cat(outs, 0), torch.cat(lang_hs, 0)

    def write(self, lang_h, ctx_h, max_words, temperature,
            stop_tokens=STOP_TOKENS, resume=False):
        """Generate a sentence word by word and feed the output of the
        previous timestep as input to the next.
        """
        # lang_h: (1, 1, 128)
        # ctx_h: (1, 1, nhid_ctx)

        outs, logprobs, lang_hs = [], [], []
        # max_words may be narrowed due to stop_tokens
        # outs: max_words*(1, 1)
        # logprobs: max_words*(1, )
        # lang_hs: (max_words+1)*(1, 128) due to the <eos> token

        # remove batch dimension from the language and context hidden states
        lang_h = lang_h.squeeze(1) # (1, 128)
        ctx_h = ctx_h.squeeze(1) # (1, nhid_ctx)

        if resume:
            inpt = None
        else:
            # if we start a new sentence, prepend it with 'YOU:'
            inpt = Variable(torch.LongTensor(1))
            inpt.data.fill_(self.word_dict.get_idx('YOU:'))
            inpt = self.to_device(inpt) # (1, )

        # generate words until max_words have been generated or <selection>
        for _ in range(max_words):
            if inpt is not None:
                # add the context to the word embedding
                inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1) # (1, nembed_word+nhid_ctx)
                # update RNN state with last word
                lang_h = self.writer(inpt_emb, lang_h) # (1, 128)
                lang_hs.append(lang_h)

            # decode words using the inverse of the word embedding matrix
            out = self.decoder(lang_h) # (1, nembed_word)
            scores = F.linear(out, self.word_encoder.weight).div(temperature) # (1, vocab_size)
            # subtract constant to avoid overflows in exponentiation
            scores = scores.add(-scores.max().item()).squeeze(0) # (vocab_size, )

            # disable special tokens from being generated in a normal turns
            if not resume:
                mask = Variable(self.special_token_mask)
                scores = scores.add(mask)

            prob = F.softmax(scores,dim=0) # (vocab_size, )
            logprob = F.log_softmax(scores,dim=0) # (vocab_size, )

            # explicitly defining num_samples for pytorch 0.4.1
            word = prob.multinomial(num_samples=1).detach() # (1, )
            logprob = logprob.gather(0, word) # (1, )

            logprobs.append(logprob)
            outs.append(word.view(word.size()[0], 1))

            inpt = word

            # check if we generated an <eos> token
            if self.word_dict.get_word(word.data[0]) in stop_tokens:
                break

        # update the hidden state with the <eos> token
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1) # (1, nembed_word+nhid_ctx)
        lang_h = self.writer(inpt_emb, lang_h) # (1, 128)
        lang_hs.append(lang_h)

        # add batch dimension back
        lang_h = lang_h.unsqueeze(1)

        return logprobs, torch.cat(outs), lang_h, torch.cat(lang_hs)
        # logprobs: max_words*(1, )
        # th.cat(outs): (max_words, 1)
        # lang_h: (1, 1, 128), hidden state of last word
        # th.cat(lang_hs): (max_words+1, 128)

    def score_sent(self, sent, lang_h, ctx_h, temperature):
        """Computes likelihood of a given sentence."""
        score = 0
        # remove batch dimension from the language and context hidden states
        lang_h = lang_h.squeeze(1)
        ctx_h = ctx_h.squeeze(1)
        inpt = Variable(torch.LongTensor(1))
        inpt.data.fill_(self.word_dict.get_idx('YOU:'))
        inpt = self.to_device(inpt)
        lang_hs = []

        for word in sent:
            # add the context to the word embedding
            inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
            # update RNN state with last word
            lang_h = self.writer(inpt_emb, lang_h)
            lang_hs.append(lang_h)

            # decode words using the inverse of the word embedding matrix
            out = self.decoder(lang_h)
            scores = F.linear(out, self.word_encoder.weight).div(temperature)
            # subtract constant to avoid overflows in exponentiation
            scores = scores.add(-scores.max().data[0]).squeeze(0)

            mask = Variable(self.special_token_mask)
            scores = scores.add(mask)

            logprob = F.log_softmax(scores)
            score += logprob[word[0]].data[0]
            inpt = Variable(word)

        # update the hidden state with the <eos> token
        inpt_emb = torch.cat([self.word_encoder(inpt), ctx_h], 1)
        lang_h = self.writer(inpt_emb, lang_h)
        lang_hs.append(lang_h)

        # add batch dimension back
        lang_h = lang_h.unsqueeze(1)

        return score, lang_h, torch.cat(lang_hs)

    def forward_context(self, ctx):
        """Run context encoder."""
        return self.ctx_encoder(ctx)

    def forward_lm(self, inpt, lang_h, ctx_h):
        """Run forward pass for language modeling."""
        # inpt: (max_len-1, <=batch_size)
        # lang_h: (1, <=batch_size, nhid_lang)
        # ctx_h: (1, <=batch_size, nhid_ctx)
        # embed words
        inpt_emb = self.word_encoder(inpt) # (max_len-1, <=batch_size, nembed_word)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.narrow(0, ctx_h.size(0) - 1, 1).expand(
            inpt.size(0), ctx_h.size(1), ctx_h.size(2)) # (max_len-1, <=batch_size, nhid_ctx)
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2) # (max_len-1, <=batch_size, nembed_word+nhid_ctx)

        inpt_emb = self.dropout(inpt_emb)

        # compact weights to reduce memory footprint
        self.reader.flatten_parameters()
        out, _ = self.reader(inpt_emb, lang_h) # (max_len-1, <=batch_size, nhid_lang)
        decoded = self.decoder(out.view(-1, out.size(2))) # (max_len-1*<=batch_size, nembed_word)

        # tie weights between word embedding/decoding
        # self.word_encoder.weight: (vocab_size, nembed_word)
        decoded = F.linear(decoded, self.word_encoder.weight) # (max_len-1*<=batch_size, vocab_size)

        return decoded.view(out.size(0), out.size(1), decoded.size(1)), out
        # return[0]: (max_len-1, <=batch_size, vocab_size)
        # return[1]: (max_len-1, <=batch_size, nhid_lang)
