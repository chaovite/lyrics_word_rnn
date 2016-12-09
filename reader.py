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

def _build_embedding(embedding_path,size):
	""" build embedding from pretrained embedding 
	To Do: enable to select just the words from the lyrics 
	instead of using the whole pre-trained embedding.
	"""
	f = open(embedding_path,'r')
	embedding_faw = f.read()
	data = np.asarry(embedding_raw.split())
	data = np.reshape(data, (-1, size + 1))
	embedding = data[:,1:size]
	words = data[:,0]
	word_to_id = dict(zip(words, range(len(words))))
	embedding = tf.convert_to_tensor(embedding, name="embedding", dtype=tf.float)
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
#   valid_path = os.path.join(data_path, "valid.txt")# remove the validation dataset
  test_path = os.path.join(data_path, "test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
#   valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, test_data


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
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size") 
      # this is to define the push a variable into the name scope.

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    # x and y are exactly offseted just by one, not by one batch!!!!
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y
