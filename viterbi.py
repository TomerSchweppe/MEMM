# !/usr/bin/env python

from features import *


def index_sentence_word(sentence, idx):
    """
    safe indexing of sentence word
    """
    if idx < 0 or idx >= len(sentence):
        return None
    return sentence[idx][0]


def index_sentence_tag(sentence, idx):
    """
    safe indexing of sentence tag
    """
    if idx < 0 or idx >= len(sentence):
        return None
    return sentence[idx][1]


class Viterbi:
    """
    viterbi class
    """

    def __init__(self, tag_list, vocab_list, v_train):
        """
        create tag-idx dictionaries
        """
        self._tag_list = tag_list
        self._tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
        self._idx_tag_dict = {idx: tag for idx, tag in enumerate(tag_list)}
        self._tags_num = len(self._tag_idx_dict)
        self._v_train = v_train

        # init feature classes
        self._f_100 = F100(vocab_list, tag_list)
        self._f_101_1 = F101(vocab_list, tag_list, 1)
        self._f_101_2 = F101(vocab_list, tag_list, 2)
        self._f_101_3 = F101(vocab_list, tag_list, 3)
        self._f_101_4 = F101(vocab_list, tag_list, 4)
        self._f_102_1 = F102(vocab_list, tag_list, 1)
        self._f_102_2 = F102(vocab_list, tag_list, 2)
        self._f_102_3 = F102(vocab_list, tag_list, 3)
        self._f_102_4 = F102(vocab_list, tag_list, 4)
        self._f_103 = F103(vocab_list, tag_list)
        self._f_104 = F104(vocab_list, tag_list)
        self._f_105 = F105(vocab_list, tag_list)

    def q(self, t_2, t_1, sentence, k):
        """
        q function
        """
        word = sentence[k]
        spr_tag_list = []
        for tag_i in self._tag_list:
            vec_list = [self._f_100(word, tag_i),
                        self._f_101_1(word, tag_i), self._f_101_2(word, tag_i), self._f_101_3(word, tag_i), self._f_101_4(word, tag_i),
                        self._f_102_1(word, tag_i), self._f_102_2(word, tag_i), self._f_102_3(word, tag_i), self._f_102_4(word, tag_i),
                        self._f_103(t_2, t_1, tag_i),
                        self._f_104(t_1, tag_i),
                        self._f_105(tag_i),
                        self._f_100(index_sentence_word(sentence, k - 1), tag_i),  # F106
                        self._f_100(index_sentence_word(sentence, k + 1), tag_i)]  # F107
            spr_tag_list.append(sparse_vec_hstack(vec_list))

        return sparse.vstack(spr_tag_list).dot(self._v_train)


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

    def run_viterbi(self, sentence):
        """
        run viterbi on sentence
        """

        # init
        n = len(sentence)
        pi = np.zeros((n, self._tags_num ** 2))
        bp = np.zeros((n, self._tags_num ** 2))
        pi[0, self.tag_pos('*', '*')] = 1

        # iterate words
        for k in range(2, n):
            # iterate u,v
            for u, v in [(x, y) for x in self._tag_list for y in self._tag_list]:
                if (u == '*' and k != 2) or u == 'STOP':
                    continue
                if v == '*':
                    continue

                values = pi[k-1,self._tag_idx_dict[u]:self._tag_idx_dict[u]+self._tags_num] + self.q(u, v, sentence, k)

                # update pi & bp
                max_pos = np.argmax(values)
                pi[k, self.tag_pos(u, v)] = values[max_pos]
                bp[k, self.tag_pos(u, v)] = max_pos


        # prediction
        pred_tag = [] * n
        (pred_tag[-2], pred_tag[-1]) = self.pos_tags(np.argmax(pi[n, :]))
        for k in range(n - 3, 1, -1):
            pred_tag[k] = self._idx_tag_dict(bp[k + 2, self.tag_pos(pred_tag[k + 1], pred_tag[k + 2])])

        return pred_tag
