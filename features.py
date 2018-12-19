#!/usr/bin/env python

import numpy as np
from scipy import sparse

class Feature:
    """
    base feature class
    """
    def __init__(self,vocab_list,tag_list):
        """save vocabulary list and tag list"""
        self._vocab_list = vocab_list
        self._tag_list = tag_list

    def __call__(self):
        """call function"""
        pass

    def feature_mat(self,start,step):
        """return numpy feature matrix"""
        mat = np.zeros([len(self._tag_list),step*len(self._tag_list)])
        if start is None:
            return mat
        for tag_idx in range(len(self._tag_list)):
            mat[tag_idx][start + step * tag_idx] = 1
        return mat

class F100(Feature):
    """
    word/tag features for all word/tag pairs
    F106 is similar -> call with previous word
    F107 is similar -> call with next word
    """
    def __init__(self, vocab_list,tag_list):
        """create word hash table"""
        super(F100,self).__init__(vocab_list,tag_list)
        self._word_idx_dict = {word: idx for idx,word in enumerate(vocab_list)}

    def __call__(self, word):
        """return F100/F106/F107 feature matrix"""
        return self.feature_mat(self._word_idx_dict.get(word,None),len(self._word_idx_dict))

class F101(Feature):
    """
    spelling features for all prefix of length n
    """
    def __init__(self, vocab_list, tag_list, n):
        """create prefix hash table"""
        super(F101,self).__init__(vocab_list, tag_list)
        assert n > 0
        self._n = n

        prefix_set = set()
        for w in self._vocab_list:
            if len(w) >= self._n:
                prefix_set.add(w[:self._n])
        self._prefix_to_idx = {prefix: idx for idx, prefix in enumerate(list(prefix_set))}

    def __call__(self, word):
        """return F101 feature matrix"""
        return self.feature_mat(self._prefix_to_idx.get(word[:self._n], None),len(self._prefix_to_idx))

class F102(Feature):
    """
    spelling features for all suffixes of length n
    """
    def __init__(self, vocab_list, tag_list, n):
        """create suffix hash table"""
        super(F102,self).__init__(vocab_list, tag_list)
        assert n > 0
        self._n = n

        suffix_set = set()
        for w in self._vocab_list:
            if len(w) >= self._n:
                suffix_set.add(w[-self._n:])
        self._suffix_to_idx = {suffix: idx for idx, suffix in enumerate(list(suffix_set))}

    def __call__(self, word):
        """return F102 feature matrix"""
        return self.feature_mat(self._suffix_to_idx.get(word[-self._n:], None),len(self._suffix_to_idx))

class F103(Feature):
    """
    2 previous & current word tag
    """
    def __init__(self, vocab_list, tag_list):
        """create tag pairs hash table"""
        super(F103, self).__init__(vocab_list, tag_list)
        self._tags_idx_dict = {(tag1,tag2): idx2+idx1*len(tag_list) for idx1, tag1 in enumerate(tag_list) for idx2, tag2 in enumerate(tag_list)}

    def __call__(self,t_2,t_1):
        "return F103 feature matrix"
        return self.feature_mat(self._tags_idx_dict.get((t_2,t_1),None),len(self._tag_list)*len(self._tag_list))

class F104(Feature):
    """
    previous & current word tag 
    """
    def __init__(self, vocab_list, tag_list):
        """create tag hash table"""
        super(F104, self).__init__(vocab_list, tag_list)
        self._tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}

    def __call__(self,t_1):
        """return F104 feature matrix"""
        return self.feature_mat(self._tag_idx_dict.get(t_1,None),len(self._tag_list))

class F105(Feature):
    """
    word tag
    """
    def __init__(self, vocab_list, tag_list):
        """init class"""
        super(F105,self).__init__(vocab_list, tag_list)

    def __call__(self):
        """return F105 feature matrix"""
        return self.feature_mat(0,1)

class Sparse_Mat(Feature):
    """
    sparse matrix class
    """
    def __init__(self,vocab_list,tag_list,word,tag,mat_list):
        """create sparse matrix"""
        super(Sparse_Mat, self).__init__(vocab_list, tag_list)
        self._word = word
        self._tag = tag
        self._spr_mat = sparse.csr_matrix(mat_list[0])
        for mat in mat_list[1:]:
            assert np.size(mat, 0) == len(self._tag_list)
            self._spr_mat = sparse.hstack([self._spr_mat,mat])

    def __call__(self):
        """return sparse matrix"""
        return self._spr_mat

    def word_tag(self):
        """return word and tag"""
        return (word,tag)

if __name__ == '__main__':
    """
    validation
    """
    vocab_list = ['tomer','ofir','roy','nadav']
    tag_list = ['S','T']

    # create all features
    n = 1
    f_100 = F100(vocab_list,tag_list)
    f_101 = F101(vocab_list,tag_list,n)
    f_102 = F102(vocab_list,tag_list,n)
    f_103 = F103(vocab_list,tag_list)
    f_104 = F104(vocab_list,tag_list)
    f_105 = F105(vocab_list,tag_list)

    # test F100
    assert np.array_equal(np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0.]]),
                                    f_100('tomer'))
    assert np.array_equal(np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0.]]),
                                    f_100('ofir'))
    assert np.array_equal(np.array([[0., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 0.]]),
                                    f_100('roy'))
    assert np.array_equal(np.array([[0., 0., 0., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 1.]]),
                                    f_100('nadav'))
    # test F101
    assert np.array_equal(f_101('tomer'), f_101('tomer' + 'not_prefix'))
    assert np.array_equal(f_101('tomer'), f_101('t'))
    assert not np.array_equal(f_101('tomer'), f_101('ofir'))

    # test F102
    assert np.array_equal(f_102('tomer'), f_102('not_suffix' + 'tomer'))
    assert np.array_equal(f_102('tomer'), f_102('r'))
    assert np.array_equal(f_102('tomer'), f_102('ofir'))

    # test F103
    assert np.array_equal(np.array([[1., 0., 0., 0.,0., 0., 0., 0.],
                                    [0., 0., 0., 0.,1., 0., 0., 0.]]),
                                    f_103('S','S'))

    assert np.array_equal(np.array([[0., 1., 0., 0.,0., 0., 0., 0.],
                                    [0., 0., 0., 0.,0., 1., 0., 0.]]),
                                    f_103('S','T'))

    assert np.array_equal(np.array([[0., 0., 1., 0.,0., 0., 0., 0.],
                                    [0., 0., 0., 0.,0., 0., 1., 0.]]),
                                    f_103('T','S'))

    assert np.array_equal(np.array([[0., 0., 0., 1.,0., 0., 0., 0.],
                                    [0., 0., 0., 0.,0., 0., 0., 1.]]),
                                    f_103('T','T'))

    # test F104
    assert np.array_equal(np.array([[1., 0., 0., 0.],
                                    [0., 0., 1., 0.]]),
                                    f_104('S'))

    assert np.array_equal(np.array([[0., 1., 0., 0.],
                                    [0., 0., 0., 1.]]),
                                    f_104('T'))
    # test F105
    assert np.array_equal(np.array([[1., 0.],
                                    [0., 1.]]),
                                    f_105())
    # test sparse matrix
    mat_list = [f_104('S'),f_105()]
    assert  np.array_equal([[1., 0., 0., 0., 1., 0.],
                           [0., 0., 1., 0., 0., 1.]],
    Sparse_Mat(vocab_list,tag_list,'tomer','S',mat_list)().todense())

    mat_list = [f_103('T','T'),f_104('T')]
    assert  np.array_equal([[0., 0., 0., 1.,0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0.,0., 0., 0., 1., 0., 0., 0., 1.]],
    Sparse_Mat(vocab_list,tag_list,'nadav','T',mat_list)().todense())

    print('PASS')
