# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("eos", "<eos>").split()

def _build_vocab(filename, embedding_folder, vocab, size, rank_top = 0, rank_low = 20000):
    """ build vocabulary from pretrained embedding 
    The words data is read from filename.
    filename: the name of the lyrics file
    if vocabulary is larger than the number of unique words from dataset, then
    the code will randomly sample the words in the pretrained embedding from rank_top
    to rank_t
    """
    if not size in {50, 100, 200, 300}:
        raise ValueError('size must be 50, 100, 200 or 300')
    
    embedding_path = os.path.join(embedding_folder,'glove.6B.{}d.txt'.format(str(size)))
    f = open(embedding_path,'r')
    embedding_raw = f.read()
    data = np.asarray(embedding_raw.split())
    data = np.reshape(data, (-1, size + 1))
    embedding_pre = data[:,1:size+1]
    embedding_pre = embedding_pre.astype(np.float32)
    words_pre = data[:,0]
    word_id_pre = dict(zip(words_pre, range(len(words_pre))))
    
    with open(filename,'r') as f:
        words_set = set(f.read().split()) # all unique words from the lyric data.
    
    words_index = set()
    for word in words_set:
        word_id = word_id_pre[word]
        words_index.add(word_id)
    
    print('{} unique words in dataset, {} words drawn from pre-trained embedding'.format(str(len(words_index)), str(vocab - len(words_index))))
    
    while len(words_index) < vocab:
        rand_index = set(np.random.random_integers(rank_top,rank_low, vocab - len(words_index)))
        words_index = words_index.union(rand_index)
    
    words_id   = np.asarray(list(words_index))
    embedding  = embedding_pre[words_id,:]
    words      = words_pre[words_id]
    word_to_id = dict(zip(words, range(len(words))))
    
    return word_to_id, embedding
    
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def lyric_raw_data(data_path=None):
    """Load raw data from data directory "data_path".
    Reads text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    Args:
    data_path: string path to the train and test data directory 

    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    total_txt_path = os.path.join(data_path,'total.txt')

    word_to_id, embedding = _build_vocab(total_txt_path, embedding_folder = 'embedding/', 
                        vocab=10000, size = 50, rank_top = 0, rank_low = 20000)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, embedding


def batch_data_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw data.

    This chunks up lyric_raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    
    Args:
      raw_data: one of the raw data outputs from raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).
    
    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.
    
    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "batch_data_producer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                        [batch_size, batch_len]) # this may throw away some data.
        
        epoch_size = (batch_len - 1) // num_steps # number of batches per unroll step
        assertion = tf.assert_positive(epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size") 
            # this is to define the push a variable into the name scope.
            
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # x and y are exactly offseted just by one, not by one batch!!!!
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x, y
