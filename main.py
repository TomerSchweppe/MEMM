#!/usr/bin/env python

import numpy as np
from features import *
from scipy import sparse
from scipy import optimize
import argparse
from multiprocessing import Pool, cpu_count
import itertools
from loss import *
import pickle
from viterbi import *

import time

RARE_THRESHOLD = 3


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
    """
    load test data
    """
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


def extract_features(vocab_list, tag_list, data, processes_num=1):
    """
    extract features from training data
    """

    # divide data into chunks
    sentence_batch_size = len(data) // processes_num
    chunks = [data[idx:idx + sentence_batch_size] for idx in range(0, len(data), sentence_batch_size)]

    # run processes
    processes = Pool()
    ret = processes.map(extract_features_thread, [(vocab_list, tag_list, chunk) for chunk in chunks])

    # combine results
    return list(itertools.chain.from_iterable(ret))


def extract_features_thread(args):
    """
    extract features from data chunk
    """
    vocab_list, tag_list, data = args

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
    spr_mats = []
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
    return spr_mats


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

def tag_pairs(data):
    """
    return tag pairs seen in data 
    """
    tag_pairs_set = set()
    for sentence in data:
        prev_tag = '*'
        for _,tag in sentence[2:]:
            tag_pairs_set.add((prev_tag,tag))
            prev_tag = tag

    return tag_pairs_set

def batch_viterbi(args):
    """
    run viterbi on sentences batch
    """
    viterbi,chunk = args
    res = []
    for sentence in chunk:
        res.append(viterbi.run_viterbi(sentence))
    return res

def parallel_viterbi(tag_list, vocab_list, v_train, train_data, test_data, processes_num):
    """
    parallel viterbi
    """
    # create viterbi class
    viterbi = Viterbi(tag_list, vocab_list, v.x, tag_pairs(train_data))

    # divide data into chunks
    sentence_batch_size = len(test_data) // processes_num
    chunks = [test_data[idx:idx + sentence_batch_size] for idx in range(0, len(test_data), sentence_batch_size)]

    processes = Pool()
    ret = processes.map(batch_viterbi, [(viterbi,chunk) for chunk in chunks])

    # combine results
    return list(itertools.chain.from_iterable(ret))


def eval(tagger, ground_truth, tag_list):
    """
    tagger evaluation
    """

    confusion_mat = np.zeros((len(tag_list),len(tag_list)),dtype=int)
    tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
    idx_tag_dict = {idx: tag for idx, tag in enumerate(tag_list)}

    for sentence in range(len(ground_truth)):
        for actual_tag, predicted_tag in zip(ground_truth[sentence],tagger[sentence]):
            if actual_tag == '*' or actual_tag == 'STOP':
                continue
            confusion_mat[tag_idx_dict[actual_tag],tag_idx_dict[predicted_tag]] += 1

    # accuracy
    print('tagger accuracy:', np.trace(confusion_mat)/np.sum(confusion_mat))
    # 10 worst in confusion matrix
    tmp = np.copy(confusion_mat)
    np.fill_diagonal(tmp, 0)
    rows = np.argsort(np.sum(tmp, axis=1))[-10:]

    # confusion matrix - 10 worst predictions
    print('confusion matrix - 10 worst predictions:')
    print('      ',end='')
    for tag in tag_list:
        print(tag, end=' ')
    print()

    row_labels = []
    for idx in rows:
        row_labels.append(idx_tag_dict[idx])
    for row_label, row in zip(row_labels, confusion_mat[rows,:]):
        print (row_label.ljust(5),end='')
        for i, val in enumerate(row):
            if i == 0:
                print(' ' + str(val), end='')
            else:
                print(' '*(len(tag_list[i-1]))+str(val),end='')
        print()


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

    # data = data[:100]

    # word & tag lists
    print('generate words and tags lists')
    vocab_list, tag_list = vocab_and_tag_lists(data)

    # remove rare words from vocabulary
    print('remove rare words from vocabulary')
    remove_rare_words(vocab_list, data)

    # extract features from training data
    print('extract features from training data')
    start = time.time()
    spr_mats = extract_features(vocab_list, tag_list, data, cpu_count())
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
    start = time.time()
    tagger = parallel_viterbi(tag_list, vocab_list, v.x, data, sentences, 4)
    print('viterbi time: ', time.time() - start)

    # evaluation
    eval(tagger,test_tags,tag_list)
