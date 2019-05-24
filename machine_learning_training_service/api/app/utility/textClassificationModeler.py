from app import app
from flask import Flask, jsonify, request
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from sklearn.externals import joblib




def textClassifierModeler():
	trainingFile = "G:\\textClassifier\\trainingData\\trainingData.json"
#training data :{"dataset":[{"patterns":["Mumbai","Chennai","Delhi"],"label":"IPL"},{"patterns":["Sydney","Melbourne","Brisbane"],"label":"BBL"},{"patterns":["Jamaica","Guyana","Barbados"],"label":"CPL"}]}
	trainedModel = "G:\\textClassifier\\trainedModel\\text_classifier.tf"	
	trainedDataPKL = "G:\\textClassifier\\trainedModel\\data.pkl"
	with open(trainingFile) as json_data:
		dataset = json.load(json_data)
	json_data.close()
	words = []
	classes = []
	documents = []
	for intent in dataset['dataset']:
		for pattern in intent['patterns']:
			w = nltk.word_tokenize(pattern)
			words.extend(w)
			documents.append((w, intent['label']))			
			if intent['label'] not in classes:
				classes.append(intent['label'])
	words = sorted(list(set(words)))
	classes = sorted(list(set(classes)))

#words : ['Barbados', 'Bombay', 'Brisbane', 'Chennai', 'Delhi', 'Guyana', 'Jamaica', 'Melbourne', 'Mumbai', 'Sydney']
#documents :[(['Mumbai'], 'IPL'), (['Chennai'], 'IPL'), (['Delhi'], 'IPL'), (['Sydney'], 'BBL'), (['Melbourne'], 'BBL'), (['Brisbane'], 'BBL'), (['Jamaica'], 'CPL'), (['Guyana'], 'CPL'), (['Barbados'], 'CPL')]
#classes :['BBL', 'CPL', 'IPL']

	training = []
	output = []
	output_empty = [0] * len(classes)
	for doc in documents:
		bag = []
		pattern_words = doc[0]
		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)

# bag :			
#[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1

# output_row :
#[0, 0, 1]
#[0, 0, 1]
#[0, 0, 1]
#[1, 0, 0]
#[1, 0, 0]
#[1, 0, 0]
#[0, 1, 0]
#[0, 1, 0]
#[0, 1, 0]
		
		training.append([bag, output_row])

		random.shuffle(training)
	training = np.array(training)

# training data:
#[[list([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) list([0, 1, 0])]
# [list([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) list([0, 0, 1])]
# [list([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) list([1, 0, 0])]
# [list([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) list([0, 1, 0])]
# [list([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) list([1, 0, 0])]
# [list([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) list([0, 0, 1])]
# [list([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) list([0, 1, 0])]
# [list([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) list([1, 0, 0])]
# [list([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) list([0, 0, 1])]]
 
	train_x = list(training[:,0])
	train_y = list(training[:,1])
	tf.reset_default_graph()
	# We are building a 3-layers neural network
	input_layer = tflearn.input_data(shape=[None, len(train_x[0])]) # preparing input layer for DNN, providing shape of the inpug data. it has 10 feeatures
	layer1 = tflearn.fully_connected(input_layer, 34)# create a fully connected layer with 34 neurons.
	layer2 = tflearn.fully_connected(layer1, 34)# create a fully connected layer with 34 neurons.
	output = tflearn.fully_connected(layer2, len(train_y[0]), activation='softmax')# create output layer 
	net = tflearn.regression(output, optimizer='Momentum', metric=tflearn.metrics.Accuracy(), loss='categorical_crossentropy')
	
	model = tflearn.DNN(net,tensorboard_dir='/app/logs/tflearn_logs')
	model.fit(train_x, train_y, n_epoch=5, batch_size=34, show_metric=True)
	model.save(trainedModel)
	joblib.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, trainedDataPKL)
	return 'training completed'
