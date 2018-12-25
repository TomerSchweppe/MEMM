#!/usr/bin/env python

import numpy as np
from features import *
from scipy import sparse
from scipy import optimize
import argparse
import os
from multiprocessing import cpu_count
from loss import *
import pickle
from viterbi import *
import sys
import time
import random


class MutePrint:
    """
    Mute Printing for the block withing this class
    """
    def __enter__(self):
        """change stdout to devnull"""
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """restore stdout to default"""
        sys.stdout = sys.__stdout__


def process_line(line):
    """process a signle tagged line"""
    return [('*', '*'), ('*', '*')] + [tuple(word_tag.split('_')) for word_tag in line.strip().split()] + [
        ('STOP', 'STOP')]


def load_data(data_file):
    """
    load training data
    """
    data = []
    with open(data_file, 'r') as fh:
        for line in fh:
            data.append(process_line(line))
    return data


def prepare_data_for_test(data):
    """prepare loaded data from load_data for testing"""
    sentences = []
    tags = []
    for tagged_sentence in data:
        sentences.append([word for (word, _) in tagged_sentence])
        tags.append([tag for (_, tag) in tagged_sentence])
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


def remove_rare_words(vocab_list, data, threshold):
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
        if vocab_dict[word] < threshold:
            vocab_list.remove(word)


def feature_vec_len(spr_mats):
    """
    return feature vector bits num
    """
    return spr_mats[0][0].shape[1]


def get_args_for_optimize(spr_mats, l):
    """
    return arguments for optimization
    """
    spr_mats_list, tag_idx_tup = zip(*spr_mats)
    spr_single_mat = sparse.vstack(spr_mats_list)
    spr_single_mat = spr_single_mat.tocsr()
    args = (spr_single_mat, tag_idx_tup, l)
    return args


def calculate_accuracy(tagger, ground_truth):
    """calculate accuracy of the tagger"""
    total = 0
    correct = 0
    for sentence in range(len(ground_truth)):
        for actual_tag, predicted_tag in zip(ground_truth[sentence], tagger[sentence]):
            if actual_tag == '*' or actual_tag == 'STOP':
                continue
            total += 1
            correct += 1 if actual_tag == predicted_tag else 0
    return correct / total



def evaluate(tagger, ground_truth, tag_list):
    """
    tagger evaluation
    """
    confusion_mat = np.zeros((len(tag_list), len(tag_list)), dtype=int)
    tag_idx_dict = {tag: idx for idx, tag in enumerate(tag_list)}
    idx_tag_dict = {idx: tag for idx, tag in enumerate(tag_list)}

    for sentence in range(len(ground_truth)):
        for actual_tag, predicted_tag in zip(ground_truth[sentence], tagger[sentence]):
            if actual_tag == '*' or actual_tag == 'STOP':
                continue
            confusion_mat[tag_idx_dict[actual_tag], tag_idx_dict[predicted_tag]] += 1

    # accuracy
    print('tagger accuracy:', np.trace(confusion_mat) / np.sum(confusion_mat))
    # 10 worst in confusion matrix
    tmp = np.copy(confusion_mat)
    np.fill_diagonal(tmp, 0)
    rows = np.argsort(np.sum(tmp, axis=1))[-10:]

    # confusion matrix - 10 worst predictions
    print('confusion matrix - 10 worst predictions:')
    print('      ', end='')
    for tag in tag_list:
        print(tag.ljust(6), end='')
    print()

    row_labels = []
    for idx in rows:
        row_labels.append(idx_tag_dict[idx])
    for row_label, row in zip(row_labels, confusion_mat[rows, :]):
        print(row_label.ljust(6), end='')
        for i, val in enumerate(row):
            print(str(val).ljust(6), end='')
        print()


def process_data_for_training(data, rare_threshold):
    """load data from train_file and process it for training"""
    # word & tag lists
    print('Generate words and tags lists')
    vocab_list, tag_list = vocab_and_tag_lists(data)

    # remove rare words from vocabulary
    print('Remove rare words from vocabulary')
    remove_rare_words(vocab_list, data, rare_threshold)

    # extract features from training data
    print('Extract features from training data')
    spr_mats = extract_features(vocab_list, tag_list, data, cpu_count())
    return vocab_list, tag_list, spr_mats


def train(data, vocab_list, tag_list, spr_mats, l):
    """train model given processed data"""
    print('Running training')
    x_0 = np.random.random(feature_vec_len(spr_mats))  # initial guess shape (n,)

    optimize_args = get_args_for_optimize(spr_mats, l)

    print('Optimize')
    opt_result = optimize.minimize(loss_function_no_for, x0=x_0, args=optimize_args, jac=dloss_dv_no_for, method='L-BFGS-B')
    return opt_result, Viterbi(tag_list, vocab_list, opt_result.x, tag_pairs(data))


def k_cross_validation(train_file, rare_threshold, k, l, beam_size):
    """perform k-fold cross validation"""
    print('Performing %d-fold Cross Validation' % k)
    data = load_data(train_file)
    random.shuffle(data)
    chunk_len = len(data) // k
    chunks = [data[i * chunk_len: (i + 1) * chunk_len] for i in range(k)]

    accuracy_accum = 0

    for i in range(k):
        # divide data to train and test sets
        train_data = chunks.copy()
        test_data = train_data.pop(i)
        train_data = list(itertools.chain.from_iterable(train_data))

        # pre-process data and train
        with MutePrint():
            vocab_list, tag_list, spr_mats = process_data_for_training(train_data, rare_threshold)
            # train on data
            result, viterbi = train(train_data, vocab_list, tag_list, spr_mats, l)
        if not result.success:
            print('in fold %d convergence failed: %s' % (i, result.message))

        # prepare data for testing, run viterbi and accumulate accuracy
        sentences, test_tags = prepare_data_for_test(test_data)
        with MutePrint():
            tagger = parallel_viterbi(viterbi, sentences, beam_size, cpu_count())
            accuracy = calculate_accuracy(tagger, test_tags)
            accuracy_accum += accuracy
        print('Fold %d Accuracy: %.3f' % (i, accuracy))

    return accuracy_accum / k


if __name__ == '__main__':
    # read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='Training set file', default='')
    parser.add_argument('--test_file', help='Test set file', default='')
    parser.add_argument('--inference_file', help='File containing sentences to tag', default='')
    parser.add_argument('--model_file', help='Trained model will be saved to this location, inference will load the '
                                             'model for this location.\n'
                                             'Exisiting model will be overridden\n'
                                             'Default is cache/model.pickle', default='cache/model.pickle')
    parser.add_argument('--rare_threshold', type=int, help='Words with term frequency under this threshold shall be '
                                                           'treated as unknowns', default=3)
    parser.add_argument('--Lambda', type=float, help='Regularization term factor for optimization', default=10 ** (-2))
    parser.add_argument('--beam_size', type=int, help='Beam size used in Viterbi for inference', default=5)
    parser.add_argument('--cross_validate', action='store_true', help='Perform cross validation to train set',
                        default=False)
    parser.add_argument('--k', type=int, help='How many folds to do in cross validation', default=7)
    args = parser.parse_args()

    to_train = False
    to_evaluate = False
    to_inference = False
    model_file = args.model_file

    if os.path.isfile(args.train_file):
        train_file = args.train_file
        to_train = True
    if os.path.isfile(args.test_file):
        test_file = args.test_file
        to_evaluate = True
    if os.path.isfile(args.inference_file):
        inference_file = args.inference_file
        to_inference = True
    if args.rare_threshold < 0:
        raise ValueError('Invalid rare_threshold = %d < 0' % args.rare_threshold)
    if args.Lambda < 0:
        raise ValueError('Invalid Lambda = %f < 0' % args.Lambda)
    if to_train and os.path.isfile(model_file):
        print('WARNING: Model at %s will be overridden' % model_file)
    if args.beam_size < 0:
        raise ValueError('Invalid beam_size = %d < 0' % args.beam_size)
    if args.k < 0:
        raise ValueError('Invalid k = %d < 0' % args.k)

    if args.cross_validate:
        if not os.path.isfile(args.train_file):
            raise FileNotFoundError('Need train_file in order to perform cross validation')
        start=time.time()
        accuracy = k_cross_validation(train_file, args.rare_threshold, args.k, args.Lambda, args.beam_size)
        print('Cross Validation Accuracy = %.3f' % accuracy)
        print('Cross Validation time: ', time.time() - start)

    if to_train:
        # load training data
        print('Loading data')
        data = load_data(train_file)

        # process training data
        start = time.time()
        vocab_list, tag_list, spr_mats = process_data_for_training(data, args.rare_threshold)
        print('Loading time: ', time.time() - start)

        # training
        start = time.time()
        result, viterbi = train(data, vocab_list, tag_list, spr_mats, args.Lambda)
        print(result)
        print('Training time: ', time.time() - start)

        # save trained model
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        pickle.dump(viterbi, open(model_file, 'wb'))

    if to_evaluate:
        if not os.path.isfile(model_file):
            raise FileNotFoundError('Saved model %s not found' % model_file)
        # test data preprocessing
        test_data = load_data(test_file)
        sentences, test_tags = prepare_data_for_test(test_data)

        # load saved model
        viterbi = pickle.load(open(model_file, 'rb'))

        # run viterbi
        start = time.time()
        tagger = parallel_viterbi(viterbi, sentences, args.beam_size, cpu_count())
        print('Viterbi time: ', time.time() - start)

        # evaluation
        start = time.time()
        evaluate(tagger, test_tags, viterbi._tag_list)
        print('Evaluation time:', time.time() - start)
