import json
import numpy as np


class UniquenessSentMetric(object):
    """Metric that evaluates the number of unique sentences."""
    def __init__(self):
        self.seen = set()

    def record(self, sen):
        self.seen.add(' '.join(sen))

    def value(self):
        return len(self.seen)


class UniquenessWordMetric(object):
    """Metric that evaluates the number of unique sentences."""
    def __init__(self):
        self.seen = set()

    def record(self, word_list):
        self.seen.update(word_list)

    def value(self):
        return len(self.seen)


def record(n_epsd, engine, N, validset, validset_stats, ppl_f, dialog, ctx_gen, rl_f, text_f):
    record_ppl(n_epsd, engine, N, validset, validset_stats, ppl_f)
    record_rl(n_epsd, dialog, ctx_gen, rl_f, text_f)


def record_ppl(n_epsd, engine, N, validset, validset_stats, ppl_f):
    loss, select_loss = engine.valid_pass(N, validset, validset_stats, rl=True)
    aver_ppl = np.exp(loss)
    ppl_f.write('{}\t{}\n'.format(n_epsd, aver_ppl))
    ppl_f.flush()
    engine.model.train()


def record_rl(n_epsd, dialog, ctx_gen, rl_f, text_f):
    conv_list = []
    reward_list = []
    sent_metric = UniquenessSentMetric()
    word_metric = UniquenessWordMetric()

    for ctxs in ctx_gen.ctxs:
        conv, agree, rewards = dialog.run(ctxs)
        true_reward = rewards[0] if agree else 0
        reward_list.append(true_reward)
        conv_list.append(conv)
        for turn in conv:
            if turn[0] == 'Alice':
                sent_metric.record(turn[1])
                word_metric.record(turn[1])

    # json.dump(conv_list, text_f, indent=4)
    aver_reward = np.average(reward_list)
    unique_sent_num = sent_metric.value()
    unique_word_num = word_metric.value()
    # TODO
    # grammar_mistake_num

    # rl_f.write('{}\t{}\t{}\t{}\t{}\n'.format(n_epsd, aver_reward, unique_sent_num, unique_word_num, grammar_mistake_num))
    rl_f.write('{}\t{}\t{}\t{}\n'.format(n_epsd, aver_reward, unique_sent_num, unique_word_num))
    rl_f.flush()
