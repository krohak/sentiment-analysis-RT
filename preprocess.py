import numpy as np

# We make use of GloVe, which converts words into vector data,
# https://nlp.stanford.edu/projects/glove/
# using the Wikipedia dataset with 50 dimensional embedding.
# https://www.damienpontifex.com/2017/10/27/using-pre-trained-glove-embeddings-in-tensorflow/

wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')

####################################

from os import listdir
from os.path import isfile, join

positiveFiles = ['txt_sentoken/pos/' + f for f in listdir('txt_sentoken/pos/') if isfile(join('txt_sentoken/pos/', f))]
negativeFiles = ['txt_sentoken/neg/' + f for f in listdir('txt_sentoken/neg/') if isfile(join('txt_sentoken/neg/', f))]
print(positiveFiles[0])
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        lines=f.readlines()
        counter = 0
        for line in lines:
            counter += len(line.split())
        numWords.append(counter)
print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        lines=f.readlines()
        counter = 0
        for line in lines:
            counter += len(line.split())
        numWords.append(counter)
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


maxSeqLength = 750


fname = positiveFiles[740]

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

#################

firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    lines=f.readlines()
    for line in lines:
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            if indexCounter < maxSeqLength:
                try:
                    firstFile[indexCounter] = wordsList.index(word)
                except ValueError:
                    firstFile[indexCounter] = 399999 #Vector for unknown words
            indexCounter = indexCounter + 1

print(firstFile)
print(indexCounter)



######################################

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')

fileCounter = 0

for pf in positiveFiles:
    with open(pf, "r") as f:
        indexCounter = 0
        lines=f.readlines()
        for line in lines:
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                print(fileCounter,indexCounter)
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            if indexCounter >= maxSeqLength:
                    break
        fileCounter = fileCounter + 1

for nf in negativeFiles:
    with open(nf, "r") as f:
        indexCounter = 0
        lines=f.readlines()
        for line in lines:
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            if indexCounter >= maxSeqLength:
                    break
        fileCounter = fileCounter + 1

np.save('idsMatrix2', ids)
