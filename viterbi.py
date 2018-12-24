# !/usr/bin/env python

from features import *
from multiprocessing import Pool
import itertools


class Viterbi:
    """
    viterbi class
    """

    def __init__(self, tag_list, vocab_list, v_train, tag_pairs):
        """
        create tag-idx dictionaries
        """
        self._tag_list = tag_list
        self._tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
        self._idx_tag_dict = {idx: tag for idx, tag in enumerate(tag_list)}
        self._tags_num = len(self._tag_idx_dict)
        self._v_train = v_train
        self._tag_pairs = tag_pairs

        # init feature classes
        self._features = Features(vocab_list, tag_list)

    def q(self, t_2, t_1, tag_i, sentence, k):
        """
        q function
        """
        word = sentence[k]
        vec_list = self._features(index_sentence_word(sentence, k - 1), word,
                                  index_sentence_word(sentence, k + 1), t_2,
                                  t_1, tag_i)
        sum = 0
        jump = 0
        for pos, window in vec_list:
            if pos >= 0:
                sum += self._v_train[pos + jump]
            jump += window

        return sum

    def tag_pos(self, x, y):
        """
        return tag pair position
        """
        return self._tag_idx_dict[x] * len(self._tag_idx_dict) + self._tag_idx_dict[y]

    def pos_tags(self, idx):
        """
        return tags from position
        """
        x_idx = idx // self._tags_num
        y_idx = idx % self._tags_num
        return self._idx_tag_dict[x_idx], self._idx_tag_dict[y_idx]

    def run_viterbi(self, sentence, beam_size=20):
        """
        run viterbi on sentence
        """
        # init
        n = len(sentence) - 1
        pi = np.full((n, self._tags_num ** 2), -np.inf)
        bp = np.zeros((n, self._tags_num ** 2))
        pi[0, self.tag_pos('*', '*')] = 0

        # iterate words
        for k in range(1, n):
            # iterate u,v
            for u, v in self._tag_pairs:
                if (u == '*' and k != 1) or u == 'STOP' or v == '*':
                    continue

                values = np.full(len(self._tag_list), -np.inf)
                for i, t in enumerate(self._tag_list):
                    if (pi[k - 1, self.tag_pos(t, u)] == -np.inf) or (t == '*' and k > 2) or (t == 'STOP'):
                        continue
                    q = self.q(t, u, v, sentence, k)
                    values[i] = pi[k - 1, self.tag_pos(t, u)] + q

                # update pi & bp
                max_pos = np.argmax(values)
                pi[k, self.tag_pos(u, v)] = values[max_pos]
                bp[k, self.tag_pos(u, v)] = max_pos
            # beam search
            pi_k = pi[k, :]
            threshold = pi_k[np.argpartition(pi_k, len(pi_k) - beam_size)[len(pi_k) - beam_size]]
            pi[k, :] = np.where(pi_k >= threshold, pi_k, -np.inf)

        # prediction
        pred_tag = ['*'] * n
        (pred_tag[-2], pred_tag[-1]) = self.pos_tags(np.argmax(pi[n - 1, :]))
        for k in range(n - 3, 1, -1):
            pred_tag[k] = self._idx_tag_dict[bp[k + 2, self.tag_pos(pred_tag[k + 1], pred_tag[k + 2])]]
        pred_tag.append('STOP')
        return pred_tag


def tag_pairs(data):
    """
    return tag pairs seen in data 
    """
    tag_pairs_set = set()
    for sentence in data:
        prev_tag = '*'
        for _, tag in sentence[2:]:
            tag_pairs_set.add((prev_tag, tag))
            prev_tag = tag

    return tag_pairs_set


def batch_viterbi(args):
    """
    run viterbi on sentences batch
    """
    viterbi, chunk = args
    res = []
    for sentence in chunk:
        res.append(viterbi.run_viterbi(sentence))
    return res


def parallel_viterbi(trained_viterbi, test_data, processes_num):
    """
    parallel viterbi
    """
    # divide data into chunks
    sentence_batch_size = len(test_data) // processes_num
    chunks = [test_data[idx:idx + sentence_batch_size] for idx in range(0, len(test_data), sentence_batch_size)]

    processes = Pool()
    ret = processes.map(batch_viterbi, [(trained_viterbi, chunk) for chunk in chunks])

    # combine results
    return list(itertools.chain.from_iterable(ret))
