###########################################################################
#
# Recurrent Neural Network Implementation from Scratch
# -- Built for a WMP KTS --
#
# Author: Sam Showalter
# Date: December 17, 2018
#
# Code and inspiration HEAVILY borrowed from:
#	https://github.com/Kulbear/deep-learning-coursera/blob/master/
#	Sequence%20Models/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step%20-%20v2.ipynb
#
# - A big and special thank you to ^, for very understandable code
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import re
import os
import datetime as dt 
import copy
import sys
import pickle as pkl

#Visualization libraries
import matplotlib.pyplot as plt

#Data Science and predictive libraries
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

#Dataset related imports
from sklearn import datasets

###########################################################################
# Helper functions 
###########################################################################

def softmax(throughput):
		"""
		Softmax function to determine pseudo probabilities for output.

		Args:

			throughput: 		Output of the feed-forward network. One-hot encoded. [num_examples x num_features]

		Returns:

			softmax probabilities as a one-hot encoded array. [num_examples x num_features]

		"""
		e_x = np.exp(throughput - np.max(throughput))
		return e_x / e_x.sum(axis = 1)[:,None]

def cross_entropy_loss(	test = False, 
						test_output = None):
    """
	Determine cross entropy (log) loss for the output. This is used to determine how fast the
	network is learning.

	Reference:
		https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays 
    
    """

    if test:
    	labels = test_output
    else:
    	labels = self.labels

    #Set the last log_loss so we can track the changes in loss
    last_log_loss = self.log_loss

    #Determine log_likelihood, convert it to loss, then update the log_loss_dif
    log_likelihood = -np.log(self.probs[np.arange(len(self.probs)),labels.argmax(axis = 1)])
    self.log_loss = np.sum(log_likelihood) / self.probs.shape[0]
    self.log_loss_dif = last_log_loss - self.log_loss_dif

###########################################################################
# Data formatting and restructuring for analysis 
###########################################################################


catch = open("C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\BA_Source_Code\\Neural_Networks\\Recurrent_Nets\\data\\catch22.txt", 'r',encoding="utf-8")
txt = catch.read()

#word_list = [i.replace("\n", "") for i in txt.split(" ")]
word_list = re.findall(r"[\w']+", txt)

print(len(set(word_list)))

print(len(set(txt)))


###########################################################################
# Class and constructor
###########################################################################

class RNN_Cell_Forward():

	def __init__(self, 
				 input_arr, 
				 hidden,
				 is_output_layer = False,

				 weights = {"W_ax": 0
				 			"W_aa": 0
				 			"W_ya": 0
				 			"b_a": 0
				 			"b_y": 0},

				 gradients = {"d_x": 0
				 			"da_prev": 0
				 			"dW_ax": 0
				 			"dW_aa": 0
				 			"d_b_a": 0}):

		self.input = input_arr
		self.hidden = hidden
		self.weights = weights
		self.gradients = gradients
		self.is_output_layer = is_output_layer

	def forward_step(self):
		self.hidden_out = np.tanh(np.dot(self.weights["W_ax"], self.weights[input_arr]) + 
								  np.dot(self.weights["W_aa"], self.hidden_in) + 
								  self.weights["b_a"])

		self.output = softmax(np.dot(self.weights["W_ya"], self.hidden_out) + 
							  self.weights["b_y"])

	def backward_step(self):
		self.



	

