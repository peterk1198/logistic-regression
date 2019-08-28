# pset6_5 matrix

import pandas as pd 
import math
import numpy as np
from tqdm import tqdm

learning_rate = .00000001
training_steps = 10000
train_file = "heart-train.txt"
test_file = "heart-test.txt"

def replace_colons(file):
	for row in file:
		row = row.replace(':', '')
		yield row

def sigmoid(z):
	return (1 / (1 + np.exp(-z)))

def ll(weights, training_examples, true_labels):
	weights *= 0
	print weights
	sum = 0
	for row in range(num_rows):
		thetaTx = np.dot(weights, training_examples[row])
		y_i = true_labels[row] 
		sum += ((y_i * np.log(sigmoid(thetaTx))) + ((1 - y_i) * np.log(1 - sigmoid(thetaTx))))
	print sum

def reformat_true_labels(labels):
	new_list = []
	for sublist in labels:
		for item in sublist:
			new_list.append(item)
	return new_list

def test_weights(weights):
	test_f_qualities = np.loadtxt(test_file, usecols = 0)
	num_features = int(test_f_qualities[0])
	num_rows = int(test_f_qualities[1])
	with open(test_file) as f:
		fixed_file = np.loadtxt(replace_colons(f), skiprows = 2)
		features_matrix = fixed_file[:, 0: (num_features)]
		true_labels = reformat_true_labels((fixed_file[:, [num_features]]))
		# creates a list to append 
		bias_x = np.ones(num_rows).T.reshape((num_rows, 1))
		# creates a matrix with 1s added to the front of each training examples for (bias)
		complete_matrix = np.concatenate([bias_x, features_matrix], axis = 1)
		# iterate over training steps
		num_correct = 0.0
		for row in range(num_rows):
			val = sigmoid(np.dot(complete_matrix[row], weights))
			if true_labels[row] == 1 and val > 0.5:
				num_correct += 1
			if true_labels[row] == 0 and val < 0.5:
				num_correct += 1
		print (num_correct / len(true_labels))

# finding the number of features and rows in the file.
tf_qualities = np.loadtxt(train_file, usecols = 0)
num_features = int(tf_qualities[0])
num_rows = int(tf_qualities[1])

with open(train_file) as f:
	# read file in, only storing matrix values.
	thetas = np.zeros(num_features + 1)
	# creates a list to append 
	bias_x = np.ones(num_rows).T.reshape((num_rows, 1))
	fixed_file = np.loadtxt(replace_colons(f), skiprows = 2)
	features_matrix = fixed_file[:, 0: (num_features)]
	true_labels = reformat_true_labels((fixed_file[:, [num_features]]))
	# creates a matrix with 1s added to the front of each training examples for (bias)
	complete_matrix = np.concatenate([bias_x, features_matrix], axis = 1)
	# iterate over training steps
	for training_step in tqdm((range(training_steps))):
		gradient = np.zeros(num_features + 1)
		theta_T_x = np.matmul(complete_matrix, thetas)
		sigmoid_theta_T_x = sigmoid(theta_T_x)
		true_labels_minus = true_labels - sigmoid_theta_T_x
		for row in range(num_rows):
			gradient += complete_matrix[row] * true_labels_minus[row]
		thetas += (learning_rate * gradient)
	test_weights(thetas)
#	ll(thetas, complete_matrix, true_labels)







