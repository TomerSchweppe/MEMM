# !/usr/bin/env python

import numpy as np
from scipy import sparse
from multiprocessing import Pool
import itertools


class Feature:
    """
    base feature class
    """

    def __init__(self, vocab_list, tag_list):
        """save vocabulary list and tag list"""
        self._vocab_list = vocab_list
        self._tag_list = tag_list
        self._word_idx_dict = {word: idx for idx, word in enumerate(vocab_list)}
        self._tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
        self._tags_num = len(tag_list)

    def __call__(self):
        """call function"""
        pass

    def feature_vec(self, shift, step, tag):
        """
        return tuple of '1' position and feature length
        """
        tag_idx = self._tag_idx_dict.get(tag, None)
        if shift is None or tag_idx is None:
            return (-1, step * self._tags_num)
        pos = shift + tag_idx * step
        return (pos, step * self._tags_num)


class F100(Feature):
    """
    word/tag features for all word/tag pairs
    F106 is similar -> call with previous word
    F107 is similar -> call with next word
    """

    def __init__(self, vocab_list, tag_list):
        """create word hash table"""
        super(F100, self).__init__(vocab_list, tag_list)

    def __call__(self, word, tag):
        """return F100/F106/F107 feature matrix"""
        return self.feature_vec(self._word_idx_dict.get(word, None), len(self._word_idx_dict), tag)


class F101(Feature):
    """
    spelling features for all prefix of length n
    """

    def __init__(self, vocab_list, tag_list, n):
        """create prefix hash table"""
        super(F101, self).__init__(vocab_list, tag_list)
        assert n > 0
        self._n = n

        prefix_set = set()
        for w in self._vocab_list:
            if len(w) >= self._n:
                prefix_set.add(w[:self._n])
        self._prefix_to_idx = {prefix: idx for idx, prefix in enumerate(list(prefix_set))}

    def __call__(self, word, tag):
        """return F101 feature matrix"""
        return self.feature_vec(self._prefix_to_idx.get(word[:self._n], None), len(self._prefix_to_idx), tag)


class F102(Feature):
    """
    spelling features for all suffixes of length n
    """

    def __init__(self, vocab_list, tag_list, n):
        """create suffix hash table"""
        super(F102, self).__init__(vocab_list, tag_list)
        assert n > 0
        self._n = n

        suffix_set = set()
        for w in self._vocab_list:
            if len(w) >= self._n:
                suffix_set.add(w[-self._n:])
        self._suffix_to_idx = {suffix: idx for idx, suffix in enumerate(list(suffix_set))}

    def __call__(self, word, tag):
        """return F102 feature matrix"""
        return self.feature_vec(self._suffix_to_idx.get(word[-self._n:], None), len(self._suffix_to_idx), tag)


class F103(Feature):
    """
    2 previous & current word tag
    """

    def __init__(self, vocab_list, tag_list):
        """create tag pairs hash table"""
        super(F103, self).__init__(vocab_list, tag_list)
        self._tags_idx_dict = {(tag1, tag2): idx2 + idx1 * self._tags_num for idx1, tag1 in enumerate(tag_list) for
                               idx2, tag2 in enumerate(tag_list)}

    def __call__(self, t_2, t_1, tag):
        "return F103 feature matrix"
        return self.feature_vec(self._tags_idx_dict.get((t_2, t_1), None), self._tags_num * self._tags_num,
                                tag)


class F104(Feature):
    """
    previous & current word tag 
    """

    def __init__(self, vocab_list, tag_list):
        """create tag hash table"""
        super(F104, self).__init__(vocab_list, tag_list)
        self._tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}

    def __call__(self, t_1, tag):
        """return F104 feature matrix"""
        return self.feature_vec(self._tag_idx_dict.get(t_1, None), self._tags_num, tag)


class F105(Feature):
    """
    word tag
    """

    def __init__(self, vocab_list, tag_list):
        """init class"""
        super(F105, self).__init__(vocab_list, tag_list)

    def __call__(self, tag):
        """return F105 feature matrix"""
        return self.feature_vec(0, 1, tag)


class StartCapital(Feature):
    """
    word starts with capital letter
    """

    def __init__(self, vocab_list, tag_list):
        """init class"""
        super(StartCapital, self).__init__(vocab_list, tag_list)

    def __call__(self, word, tag):
        """
        start with capital letter / all capital letters
        """
        if word[0].isupper:
            return self.feature_vec(0, 1, tag)
        return self.feature_vec(None, 1, tag)


class AllCapital(Feature):
    """
    word contain only capital letters
    """

    def __init__(self, vocab_list, tag_list):
        """init class"""
        super(AllCapital, self).__init__(vocab_list, tag_list)

    def __call__(self, word, tag):
        """
        start with capital letter / all capital letters
        """
        if word.isupper:
            return self.feature_vec(0, 1, tag)
        return self.feature_vec(None, 1, tag)


class Number(Feature):
    """
    word is a number
    """

    def __init__(self, vocab_list, tag_list):
        """init class"""
        super(Number, self).__init__(vocab_list, tag_list)

    def __call__(self, word, tag):
        """
        start with capital letter / all capital letters
        """
        if word.isdigit():
            return self.feature_vec(0, 1, tag)
        return self.feature_vec(None, 1, tag)


class Features():
    """
    Features class
    """

    def __init__(self, vocab_list, tag_list):
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
        self._start_capital = StartCapital(vocab_list, tag_list)
        self._all_capital = AllCapital(vocab_list, tag_list)
        self._number = Number(vocab_list, tag_list)

    def __call__(self, prev_word, word, next_word, tag_2, tag_1, tag_i):
        """
        return list of all features
        """
        return [self._f_100(word, tag_i),
                self._f_101_1(word, tag_i), self._f_101_2(word, tag_i), self._f_101_3(word, tag_i),
                self._f_101_4(word, tag_i),
                self._f_102_1(word, tag_i), self._f_102_2(word, tag_i), self._f_102_3(word, tag_i),
                self._f_102_4(word, tag_i),
                self._f_103(tag_2, tag_1, tag_i),
                self._f_104(tag_1, tag_i),
                self._f_105(tag_i),
                self._f_100(prev_word, tag_i),  # F106
                self._f_100(next_word, tag_i),  # F107
                self._start_capital(word, tag_i),
                self._all_capital(word, tag_i),
                self._number(word, tag_i)]


def spr_feature_vec(vec_list):
    """
    return sparse vector of vector list 
    """
    jump = 0
    col = []
    data_length = 0
    for pos, window in vec_list:
        if pos >= 0:
            col.append(pos + jump)
            data_length += 1
        jump += window
    data = np.ones(data_length, dtype=bool)

    return sparse.coo_matrix((data, ([0] * len(col), col)), shape=(1, jump), dtype=bool)


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


def extract_features(vocab_list, tag_list, data, processes_num):
    """
    extract features from training data
    """

    # divide data into chunks
    sentence_batch_size = len(data) // processes_num
    chunks = [data[idx:idx + sentence_batch_size] for idx in range(0, len(data), sentence_batch_size)]

    # init features
    features = Features(vocab_list, tag_list)

    # run processes
    processes = Pool()
    ret = processes.map(extract_features_thread, [(features, vocab_list, tag_list, chunk) for chunk in chunks])

    # combine results
    return list(itertools.chain.from_iterable(ret))


def extract_features_thread(args):
    """
    extract features from data chunk
    """
    features, vocab_list, tag_list, data = args

    tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
    spr_mats = []
    # collect sparse matrices for each word/tag pair
    for sentence in data:
        for idx, (word, tag) in enumerate(sentence):
            spr_tag_list = []

            for tag_i in tag_list:
                vec_list = features(index_sentence_word(sentence, idx - 1), word,
                                    index_sentence_word(sentence, idx + 1), index_sentence_tag(sentence, idx - 2),
                                    index_sentence_tag(sentence, idx - 1), tag_i)
                spr_tag_list.append(spr_feature_vec(vec_list))

            spr_mats.append((sparse.vstack(spr_tag_list), tag_idx_dict[tag]))
    return spr_mats


if __name__ == '__main__':
    """
    validation
    """
    vocab_list = ['tomer', 'ofir', 'roy', 'nadav']
    tag_list = ['S', 'T']

    # create all features
    n = 1
    f_100 = F100(vocab_list, tag_list)
    f_101 = F101(vocab_list, tag_list, n)
    f_102 = F102(vocab_list, tag_list, n)
    f_103 = F103(vocab_list, tag_list)
    f_104 = F104(vocab_list, tag_list)
    f_105 = F105(vocab_list, tag_list)
    start_capital = StartCapital(vocab_list, tag_list)
    all_capital = AllCapital(vocab_list, tag_list)
    number = Number(vocab_list, tag_list)

    # test F100
    assert np.array_equal(np.array([[1, 0, 0, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_100('tomer', 'S')]).todense())
    assert np.array_equal(np.array([[0, 1, 0, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_100('ofir', 'S')]).todense())
    assert np.array_equal(np.array([[0, 0, 0, 0, 0, 0, 1, 0]]),
                          spr_feature_vec([f_100('roy', 'T')]).todense())
    assert np.array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 1]]),
                          spr_feature_vec([f_100('nadav', 'T')]).todense())
    assert np.array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_100('not_in_vocab', 'T')]).todense())

    # # test F101
    assert np.array_equal(spr_feature_vec([f_101('tomer', 'S')]).todense(),
                          spr_feature_vec([f_101('tomer' + 'not_prefix', 'S')]).todense())
    assert np.array_equal(spr_feature_vec([f_101('tomer', 'S')]).todense(),
                          spr_feature_vec([f_101('t', 'S')]).todense())
    assert not np.array_equal(spr_feature_vec([f_101('tomer', 'S')]).todense(),
                              spr_feature_vec([f_101('t', 'T')]).todense())
    assert not np.array_equal(spr_feature_vec([f_101('tomer', 'S')]).todense(),
                              spr_feature_vec([f_101('ofir', 'S')]).todense())
    # # test F102
    assert np.array_equal(spr_feature_vec([f_102('tomer', 'S')]).todense(),
                          spr_feature_vec([f_102('not_suffix' + 'tomer', 'S')]).todense())
    assert np.array_equal(spr_feature_vec([f_102('tomer', 'S')]).todense(),
                          spr_feature_vec([f_102('r', 'S')]).todense())
    assert np.array_equal(spr_feature_vec([f_102('tomer', 'S')]).todense(),
                          spr_feature_vec([f_102('ofir', 'S')]).todense())
    # test F103
    assert np.array_equal(np.array([[1, 0, 0, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_103('S', 'S', 'S')]).todense())
    assert np.array_equal(np.array([[0, 1, 0, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_103('S', 'T', 'S')]).todense())
    assert np.array_equal(np.array([[0, 0, 1, 0, 0, 0, 0, 0]]),
                          spr_feature_vec([f_103('T', 'S', 'S')]).todense())
    assert np.array_equal(np.array([[0, 0, 0, 1, 0, 0, 0, 0]]),
                          spr_feature_vec([f_103('T', 'T', 'S')]).todense())
    # # test F104
    assert np.array_equal(np.array([[1, 0, 0, 0]]),
                          spr_feature_vec([f_104('S', 'S')]).todense())
    assert np.array_equal(np.array([[0, 1, 0, 0]]),
                          spr_feature_vec([f_104('T', 'S')]).todense())
    # # test F105
    assert np.array_equal(np.array([[1, 0]]),
                          spr_feature_vec([f_105('S')]).todense())
    assert np.array_equal(np.array([[0, 1]]),
                          spr_feature_vec([f_105('T')]).todense())

    # test sparse vector
    vec_list = [f_103('T', 'T', 'S'), f_105('S'), f_100('roy', 'T')]
    assert np.array_equal([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                          spr_feature_vec(vec_list).todense())

    print('PASS')
