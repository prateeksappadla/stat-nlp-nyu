import gensim
import os
import sys

# Read the input data and convert it into list of sentences
inputDirectory = '/Users/prateek/Downloads/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/'
sentences = []  
for filename in os.listdir(inputDirectory):
    for line in open(os.path.join(inputDirectory,filename)):
	sentences.append(line.split())

# The parameters of the word2vec model
embeddingSize = 100
windowSize = 10
minCount = 5
negativeSampleSize = 5
epochs = 10

# Learn the embeddings
model = gensim.models.Word2Vec(sentences,size=embeddingSize, window=windowSize, min_count=minCount, negative=negativeSampleSize, iter=epochs, workers=3)

# Save the learned embeddings
outputPath ='/Users/prateek/Documents/Fall2016Acads/StatNLP/data5/training-data/'
outputFilename = 'NgensimBillion_' + 'embSize%s_' % embeddingSize + 'window%s' % windowSize 
model.save_word2vec_format(outputPath + outputFilename, binary=False)

	
