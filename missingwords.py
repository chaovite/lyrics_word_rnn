# this script print all the missing words in the volcabulary that are in the lyrics.
import txt_prep
import numpy as np

# load lyrics data
with open('data/total.txt','r') as f:
    words = f.read().split()

# load embedding
size = 50
emd_path = 'embedding/glove.6B.50d.txt'
with open(emd_path,'r') as f:
    print('loading embedding....')
    data = np.asarray(f.read().split())
    data = np.reshape(data,(-1, size + 1))
    word_id = dict(zip(data[:,0], range(len(data[:,0]))))
    print('Finishing constructing embedding and word_id')

words_miss = set()
for word in words:
    if not word in word_id:
        words_miss.add(word)

print('%d words missing in the vocabulary of the pretrained embedding'%(len(words_miss)))
print(words_miss)
