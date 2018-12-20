#!/usr/bin/env python

import numpy as np
from features import *
from scipy import sparse
import argparse
from threading import Thread
import itertools

import time

RARE_THRESHOLD = 3
LAMBDA = 0.1

def load_data(train_file):
    """
    load training data
    """
    data = []
    with open(train_file,'r') as fh:
        for line in fh:
            data.append([tuple(word_tag.split('_')) for word_tag in line.strip().split()])
    return data

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
    return list(word_set),list(tag_set)

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

def extract_features(vocab_list,tag_list,data,threads):
    """
    extract features from training data
    """

    data = data[:8]

    # divide data into chunks
    sentence_batch_size = len(data)//threads
    chunks = [data[idx:idx + sentence_batch_size] for idx in range(0, len(data), sentence_batch_size)]
    threads = []
    spr_mats = [[] for _ in range(len(chunks))]

    # run threads
    for idx,chunk in enumerate(chunks):
        thread = Thread(target=extract_features_thread, args=(vocab_list, tag_list, chunk, spr_mats[idx]))
        thread.start()
        threads.append(thread)

    # wait for threads to finish
    for thread in threads:
        thread.join()

    # combine results
    return list(itertools.chain.from_iterable(spr_mats))

def extract_features_thread(vocab_list,tag_list,data,spr_mats):
    """
    extract features from data chunk
    """

    # init feature classes
    f_100 = F100(vocab_list, tag_list)
    f_101_1 = F101(vocab_list,tag_list,  1)
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
                vec_list = [f_100(word,tag_i),
                        f_101_1(word,tag_i), f_101_2(word,tag_i), f_101_3(word,tag_i), f_101_4(word,tag_i),
                        f_102_1(word,tag_i), f_102_2(word,tag_i), f_102_3(word,tag_i), f_102_4(word,tag_i),
                        f_103(index_sentence_tag(sentence, idx - 2), index_sentence_tag(sentence, idx - 1),tag_i),
                        f_104(index_sentence_tag(sentence, idx - 1),tag_i),
                        f_105(tag_i),
                        f_100(index_sentence_word(sentence, idx - 1),tag_i), # F106
                        f_100(index_sentence_word(sentence, idx + 1),tag_i)] # F107
                spr_tag_list.append(sparse_vec_hstack(vec_list))
            spr_mats.append((sparse.vstack(spr_tag_list),tag_idx_dict[tag]))


def rare(vocab_list,data):
    """
    remove rare words from vocab list
    """
    vocab_dict = {}

    for sentence in data:
        for (word, _) in sentence:
            if word in vocab_dict:
                vocab_dict[word]+=1
            else:
                vocab_dict[word]=0

    for word in vocab_dict:
        if vocab_dict[word] < RARE_THRESHOLD:
            vocab_list.remove(word)

if __name__ == '__main__':
    # read input arguments
    train_file = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='training set file',default='train.wtag')
    args = parser.parse_args()

    if args.train_file:
        train_file = args.train_file

    # load training data
    print('loading data')
    data = load_data(train_file)

    # word & tag lists
    print('generate words and tags lists')
    vocab_list, tag_list = vocab_and_tag_lists(data)

    # remove rare words from vocabulary
    print('remove rare words from vocabulary')
    rare(vocab_list,data)

    # extract features from training data
    print('extract features from training data')
    start = time.time()
    spr_arr = extract_features(vocab_list,tag_list,data,8)
    print('extract time: ',time.time()-start)


    print('loss: ',loss_function(spr_arr,spr_arr))