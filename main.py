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

import time


def process_line(line,delimiter):
    """process a signle tagged line"""
    return [('*', '*'), ('*', '*')] + [tuple(word_tag.split(delimiter)) for word_tag in line.strip().split()] + [
        ('STOP', 'STOP')]



def load_data(data_file):
    """
    load training data
    """
    data = []
    with open(data_file, 'r') as fh:
        for line in fh:
                data.append(process_line(line,'_'))
    return data


def load_test(test_file,comp=False):
    """
    load test data
    """
    sentences = []
    tags = []
    with open(test_file, 'r') as fh:
        for line in fh:
            if comp:
                word_tag = (process_line(line,' '))
                sentences.append([word[0] for word in word_tag])
            else:
                word_tag = (process_line(line,'_'))
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
    print('     ', end='')
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


# def k_cross_validation(train_file, rare_threshold, k, l):
#     """perform k-fold cross validation"""
#     print('Performing %d-fold Cross Validation' % k)
#     data = load_data(train_file)
#     chunk_len = len(data[:100]) // k
#     chunks = [data[i * chunk_len: (i + 1) * chunk_len] for i in range(k)]
#
#     accuracy_accum = 0
#
#     for i in range(k):
#         # divide data to train and test sets
#         train_data = chunks.copy()
#         test_data = train_data.pop(i)
#
#         # pre-process data
#         vocab_list, tag_list, spr_mats = process_data_for_training(data, args.rare_threshold)
#
#         # train on data
#         result, viterbi = train(train_data, vocab_list, tag_list, spr_mats, l)
#         if not result.success:
#             print('in fold %d convergence failed: %s' % (i, result.message))






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
    parser.add_argument('--k', type=int, help='How many folds to do in cross validation', default=10)
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

    # if args.cross_validate:
    #     if not os.path.isfile(args.train_file):
    #         raise FileNotFoundError('Need train_file in order to perform cross validation')
    #     start=time.time()
    #     k_cross_validation(train_file, args.rare_threshold, args.k, args.Lambda)
    #     print('Cross Validation time: ', time.time() - start)

    if to_train:
        # load training data
        print('Loading data')
        data = load_data(train_file)

        data = data[:100]

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
        sentences, test_tags = load_test(test_file)

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

    # create competition file
    if to_inference:
        sentences, _ = load_test(inference_file, comp=True)
        viterbi = pickle.load(open(model_file, 'rb'))
        tagger = parallel_viterbi(viterbi, sentences, args.beam_size, cpu_count())
        if inference_file == 'comp.words':
            f_name = 'comp_m1_203764618.wtag'
        else:
            f_name = 'comp_m2_203764618.wtag'
        fh = open(f_name,'w')
        for sen_idx,sentence in enumerate(sentences):
            for word,tag in zip(sentence,tagger[sen_idx]):
                if word != '*' and word != 'STOP':
                    fh.write(word+'_'+tag+' ')
        fh.close()




