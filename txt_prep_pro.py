# This module is to generate training, testing and validation data
# from postprocessed lyrics data.

#""" This function reads in post-processed lyric data and return
#    train, valid, test dataset. train_ratio and valid ratio is ratio 
#    of data used for training and validation purposes, the rest of 
#    the data is used for test purposes.
#"""

import numpy as np

file = 'data/Lyrics_processed.txt'
train_ratio = 0.8
valid_ratio = 0.1
save_folder = 'data/'

count = 0
Lyrics = []
with open(file,'r') as f:
    lines = f.readlines()
    for line in lines:
        count += 1
        if count % 2 == 0:
            # throw away lyrics with only instrumental
            if line[0:12] !='instrumental':
                Lyrics.append(line.replace('\n',' ')) #replace end of line with white space
ndata  = len(Lyrics)
ntrain = round(ndata*train_ratio) + 1
nvalid = round(ndata*valid_ratio) + 1
order = np.arange(ndata)
np.random.shuffle(order)# shuffle the order;
index_train = order[0:ntrain]
index_valid = order[ntrain:ntrain + nvalid]
index_test  = order[ntrain + nvalid:]
# write train, valid and test txt files.
with open(save_folder+'train.txt','w') as f_train:
    for ind in index_train:
        f_train.write(Lyrics[ind])
        f_train.write(' eos ') 
with open(save_folder+'valid.txt','w') as f_valid:
    for ind in index_valid:
        f_valid.write(Lyrics[ind])
        f_valid.write(' eos ') 
with open(save_folder+'test.txt','w') as f_test:
    for ind in index_test:
        f_test.write(Lyrics[ind])
        f_test.write(' eos ')
with open(save_folder+'total.txt','w') as f_total:
    for lyric in Lyrics:
        f_total.write(lyric)
        f_total.write(' eos ')
print('Done with writing train, valid and test data!')

