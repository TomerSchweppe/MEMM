#!/usr/bin/env python

import numpy as np
from features import *
from scipy import sparse
from scipy import optimize
import argparse
from threading import Thread
import itertools
from loss import *
from viterbi import *

import time

RARE_THRESHOLD = 5


def load_data(train_file):
    """
    load training data
    """
    data = []
    with open(train_file, 'r') as fh:
        for line in fh:
            data.append([('*', '*'), ('*', '*')] + [tuple(word_tag.split('_')) for word_tag in line.strip().split()] + [
                ('STOP', 'STOP')])
    return data


def load_test(test_file):
    sentences = []
    tags = []
    with open(train_file, 'r') as fh:
        for line in fh:
            word_tag = ([('*', '*'), ('*', '*')] + [tuple(word_tag.split('_')) for word_tag in line.strip().split()] + [
                ('STOP', 'STOP')])
            sentences.append([word for (word, _) in word_tag])
            tags.append([tag for (_, tag) in word_tag])

    return sentences, tags


def vocab_and_tag_lists(data):
    """
    generate word and tag dictionaries
    """
    word_set = set()
    tag_set = set()
    for sentence in data:
        for word_tag_pair in sentence:
            word_set.add(word_tag_pair[0])
            tag_set.add(word_tag_pair[1])
    return list(word_set), list(tag_set)


def extract_features(vocab_list, tag_list, data, threads=1):
    """
    extract features from training data
    """

    data = data[:100]

    # Multithreading hurts performance!!!
    threads = 1

    # divide data into chunks
    sentence_batch_size = len(data) // threads
    chunks = [data[idx:idx + sentence_batch_size] for idx in range(0, len(data), sentence_batch_size)]
    threads = []
    spr_mats = [[] for _ in range(len(chunks))]

    # run threads
    for idx, chunk in enumerate(chunks):
        thread = Thread(target=extract_features_thread, args=(vocab_list, tag_list, chunk, spr_mats[idx]))
        thread.start()
        threads.append(thread)

    # wait for threads to finish
    for thread in threads:
        thread.join()

    # combine results
    return list(itertools.chain.from_iterable(spr_mats))


def extract_features_thread(vocab_list, tag_list, data, spr_mats):
    """
    extract features from data chunk
    """
    # init feature classes
    f_100 = F100(vocab_list, tag_list)
    f_101_1 = F101(vocab_list, tag_list, 1)
    f_101_2 = F101(vocab_list, tag_list, 2)
    f_101_3 = F101(vocab_list, tag_list, 3)
    f_101_4 = F101(vocab_list, tag_list, 4)
    f_102_1 = F102(vocab_list, tag_list, 1)
    f_102_2 = F102(vocab_list, tag_list, 2)
    f_102_3 = F102(vocab_list, tag_list, 3)
    f_102_4 = F102(vocab_list, tag_list, 4)
    f_103 = F103(vocab_list, tag_list)
    f_104 = F104(vocab_list, tag_list)
    f_105 = F105(vocab_list, tag_list)

    tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}

    # collect sparse matrices for each word/tag pair
    for sentence in data:
        for idx, (word, tag) in enumerate(sentence):
            spr_tag_list = []

            for tag_i in tag_list:
                vec_list = [f_100(word, tag_i),
                            f_101_1(word, tag_i), f_101_2(word, tag_i), f_101_3(word, tag_i), f_101_4(word, tag_i),
                            f_102_1(word, tag_i), f_102_2(word, tag_i), f_102_3(word, tag_i), f_102_4(word, tag_i),
                            f_103(index_sentence_tag(sentence, idx - 2), index_sentence_tag(sentence, idx - 1), tag_i),
                            f_104(index_sentence_tag(sentence, idx - 1), tag_i),
                            f_105(tag_i),
                            f_100(index_sentence_word(sentence, idx - 1), tag_i),  # F106
                            f_100(index_sentence_word(sentence, idx + 1), tag_i)]  # F107

                spr_tag_list.append(spr_feature_vec(vec_list))

            spr_mats.append((sparse.vstack(spr_tag_list), tag_idx_dict[tag]))


def remove_rare_words(vocab_list, data):
    """
    remove rare words from vocab list
    """
    vocab_dict = {}

    for sentence in data:
        for (word, _) in sentence:
            if word in vocab_dict:
                vocab_dict[word] += 1
            else:
                vocab_dict[word] = 0

    for word in vocab_dict:
        if vocab_dict[word] < RARE_THRESHOLD:
            vocab_list.remove(word)


def feature_vec_len(spr_mats):
    """
    return feature vector bits num
    """
    return spr_mats[0][0].shape[1]


def get_args_for_optimize(spr_mats):
    """
    return arguments for optimization
    """
    spr_mats_list, tag_idx_tup = zip(*spr_mats)
    spr_single_mat = sparse.vstack(spr_mats_list)
    spr_single_mat = spr_single_mat.tocsr()
    args = (spr_single_mat, tag_idx_tup)
    return args


if __name__ == '__main__':
    # read input arguments
    train_file = None
    test_file = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='training set file', default='train.wtag')
    parser.add_argument('--test_file', help='test set file', default='test.wtag')
    args = parser.parse_args()

    if args.train_file:
        train_file = args.train_file
    if args.test_file:
        test_file = args.test_file

    # load training data
    print('loading data')
    data = load_data(train_file)

    # word & tag lists
    print('generate words and tags lists')
    vocab_list, tag_list = vocab_and_tag_lists(data)

    # remove rare words from vocabulary
    print('remove rare words from vocabulary')
    remove_rare_words(vocab_list, data)

    # extract features from training data
    print('extract features from training data')
    start = time.time()
    spr_mats = extract_features(vocab_list, tag_list, data, 8)
    print('extract time: ', time.time() - start)

    # training
    print('running training')
    start = time.time()
    x_0 = np.random.random(feature_vec_len(spr_mats))  # initial guess shape (n,)

    args = get_args_for_optimize(spr_mats)

    print('optimize')
    v = optimize.minimize(loss_function_no_for, x0=x_0, args=args, jac=dloss_dv_no_for, method='L-BFGS-B')
    print('training time: ', time.time() - start)
    print(v)

    # test data preprocessing
    sentences, test_tags = load_test(test_file)

    # run viterbi
    print('running viterbi')
    viterbi = Viterbi(tag_list, vocab_list, v.x)
    start = time.time()
    print(viterbi.run_viterbi(sentences[0]))
    print(test_tags[0])
    print('viterbi time: ', time.time() - start)
