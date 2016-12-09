# this script print all the missing words in the volcabulary that are in the lyrics.
import txt_prep
import numpy as np
import json


#Lyric_processed = txt_prep.train_test_txt_gen(json_file='data/LyricsData.json',train_portion=0.8,save_folder ='data/')

# load processed lyrics data
filename = 'words_miss/All_Lyrics_fixed.txt'
#filename  = 'data/total.txt'
with open(filename,'r') as f:
    Lyrics_All = f.read()
    words = Lyrics_All.split()
    
# load the unprocessed lyrics data
print('There are {} unique words in the dataset'.format(len(set(words))))
# load embedding
size = 50
emd_path = 'embedding/glove.6B.50d.txt'
with open(emd_path,'r') as f:
    print('loading embedding....')
    data = np.asarray(f.read().split())
    data = np.reshape(data,(-1, size + 1))
    with open('data/word_id.txt','w') as f_id:
        for i in range(len(data[:,0])):
            f_id.write('{}: {}\n'.format(str(i+1),data[i,0]))

    word_id = dict(zip(data[:,0], range(len(data[:,0]))))
    print('Finishing constructing embedding and word_id')

#all the words missing
words_miss = set()
for word in words:
    if not word in word_id:
        words_miss.add(word)

print('Processed lyrics: %d words missing in the vocabulary of the pretrained embedding'%(len(words_miss)))
print(words_miss)

# output all the missing words:
count = 0
with open('words_miss/all_miss_words.txt','w') as f:
    for word in words_miss:
        count = count + 1
        f.write('{}: {} \n'.format(str(count), word))

print('Removing all missing words are deleted from the dataset!')

with open(filename,'r') as f:
    lines = f.readlines()

# remove missing words line by line
words_new = []
newlines =  []
for line in lines:
    newline =''
    words = line.split()
    for word in words:
        if word in word_id:
            newline = newline + ' '+ word
    newline = newline+' \n'
    newlines.append(newline)
    words = newline.split()
    for word in words:
        words_new.append(word)

# check if all the missing words have been removed!
words_miss_new = set()
for word in words_new:
    if not word in word_id:
        words_miss_new.add(word)

print('Processed lyrics: %d words missing in the vocabulary of the pretrained embedding'%(len(words_miss_new)))
print(words_miss_new)
if not words_miss_new:
    print('All the missing words are indeed removed!')
    print('Writing to data/Lyrics_processed.txt')
    with open ('data/Lyrics_processed.txt','w') as f:
        for line in newlines:
             line = line.lower()
             f.write(line)

# delete all missing words from the lyrics data and save it to a file.
#with open('data/Lyrics_processed.txt','w') as f:
#    f.write(Lyrics_All)

##missing words per lyric.
#words_miss_per_lyric = []
#for lyric in Lyric_processed:
#    word_miss_set = set()
#    lyric_words = lyric.split()
#    for word in lyric_words:
#        if not word in word_id:
#            word_miss_set.add(word)
#    words_miss_per_lyric.append(word_miss_set)
#
## print all words and word id
#with open('words_miss/word_id.txt','w') as f:
#    for key in word_id:
#        print_str = key + ' '+ str(word_id[key]) + '\n'
#        f.write(print_str)
#
## print missing words with lyric number:
#count = 0
#with open('words_miss/words_miss_per_lyric.txt','w') as f:
#    for word_miss_set in words_miss_per_lyric:
#        count = count + 1
#        f.write('Lyric {} has {} words missing from embedding vocabulary \n'.format(str(count),str(len(word_miss_set))))
#        if word_miss_set:
#            f.write('These words are: ')
#            for word in word_miss_set:
#                f.write(word + ' ')
#            f.write('\n')
#
## print the lyrics of all the processed
#count = 0
#with open('words_miss/All_Lyrics.txt','w') as f:
#    for lyric in Lyric_processed:
#        count = count + 1
#        f.write(str(count)+'\n')
#        f.write(lyric + '\n')

