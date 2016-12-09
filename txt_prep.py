#This script pre process the lyrics data stored in json file
# and then output training and testing txt file.
import json
import os
import string   
import numpy as np

def train_test_txt_gen(json_file='data/LyricsData.json',train_portion = 0.8, save_folder = 'data/'):
    """ load lyrics data from json file and divide the two portions.
    train_percent of all lyrics data will be used for training while
    the rest of the data will be used for testing. 
    The code will seperate punctuations as seperated word with white space.
    Input args:
    json_file: the path of the lyric json file.
    train_portion: the percentage of the lyrics used for training 
    (the rest will be used for testing.)
    save_folder: the folder where train.txt and test.txt will be stored.
    """
    f = open(json_file,'r')
    data = json.load(f)
    f.close()
    Lyrics = []
    
    for lyric in data['Lyrics']:
        if len(lyric) > 5: # squeeze empty lyrics out.
            lyric = lyric.replace('\r',' ') # remove \r
            lyric = lyric.replace('\n',' eos ') # replace \n with EOS
            lyric = lyric.lower()# change all words to lower case.
            Lyrics.append(lyric)

    punctuation = string.punctuation
    # add a special punctuation (the following line is hard coded!)
    p_special = '’'+'‘' # '’'
    print(p_special)
    # replace '’' with "'"
    for i in range(len(Lyrics)):
        Lyrics[i] = Lyrics[i].replace('’',"'")
        Lyrics[i] = Lyrics[i].replace('‘',"'")
        Lyrics[i] = Lyrics[i].replace('“',"'")
        Lyrics[i] = Lyrics[i].replace('”',"'")
    # add white space to the front of every punctuation;
    Lyrics_process = [];
    count = 0   
    for lyric in Lyrics:
        count = count + 1
        print('%d th song in %d songs' %(count, len(Lyrics)))
        lyric_p = ''
        for i in range(len(lyric)):
            c = lyric[i]
            if c in punctuation:
                # read until white space;
                chars = ''
                for c_p in lyric[i+1:]:
                    if c_p != ' ':
                        chars = chars + c_p
                    else:
                        break            
                if chars in {'s','ve','d','re','m','ll'}:
                    lyric_p = lyric_p + ' '+ c # add one space
                elif chars == 't' and lyric[i-1] == 'n': # don't split n't
                    lyric_p = lyric_p + c
                else:
                    lyric_p = lyric_p + ' '+ c + ' '
            else:
                lyric_p = lyric_p + c
    # replace n't with ' n't'
        lyric_p = lyric_p.replace("n't"," n't")
        Lyrics_process.append(lyric_p)
    print('Finish Lyrics Preprocessing!')
    ntotal = len(Lyrics_process) # total number of lyrics
    ntrain = round(ntotal * train_portion) + 1 # number of training samples
    # shuffle the order of lyrics.
    order = np.arange(ntotal)
    np.random.shuffle(order)# shuffle the order;
    index_train = order[0:ntrain]
    index_test  = order[ntrain:ntotal]
    with open(save_folder+'train.txt','w') as f_train:
        for ind in index_train:
            f_train.write(Lyrics_process[ind])
            f_train.write(' eos ')
    with open(save_folder+'test.txt','w') as f_test:
        for ind in index_test:
            f_test.write(Lyrics_process[ind])
            f_test.write(' eos ')
    with open(save_folder+'total.txt','w') as f_total:
        for lyric in Lyrics_process:
            f_total.write(lyric)
            f_total.write(' eos ')
    print('Finish writing training, test and total text files to save folder')
    return Lyrics_process

# running this script
#Lyric_processed = txt_prep.train_test_txt_gen(json_file='data/LyricsData.json',save_folder ='data/')
