###########################################################################
#
# Neural Network Implementation from Scratch
# -- Built for a WMP KTS --
#
# Author: Sam Showalter
# Date: October 11, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import os
import datetime as dt 
import copy
import sys
import pickle as pkl
import re

#Visualization libraries
import matplotlib.pyplot as plt

#Data Science and predictive libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

#Deep learning libraries import
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#Dataset related imports
from sklearn import datasets


###########################################################################
# Data formatting and restructuring for analysis 
###########################################################################

catch = open("C:\\Users\\sshowalter\\Desktop\\ExcludeFromBackup\\Repos\\Recurrent_Story_Telling\\data\\catch22.txt", 'r',encoding="utf-8")

raw_text = catch.read().lower()

#word_list = [i.replace("\n", "") for i in txt.split(" ")]
#word_list = re.findall(r"[\w']+", txt)

#set_words = set(word_list)

char_to_int = dict((c, i) for i, c in enumerate(set(raw_text)))

n_chars = len(raw_text)
n_vocab = len(char_to_int)

print(n_chars,n_vocab)

#Chunk up the sequence
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)



# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

print(y[0:10])


###########################################################################
# Model Implementation
###########################################################################



# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

os.chdir("C:\\Users\\sshowalter\\Desktop\\ExcludeFromBackup\\Repos\\Recurrent_Story_Telling\\weights")
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, 
	monitor='loss', 
	verbose=1, 
	save_best_only=True, 
	mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=5, batch_size=128, callbacks=callbacks_list)