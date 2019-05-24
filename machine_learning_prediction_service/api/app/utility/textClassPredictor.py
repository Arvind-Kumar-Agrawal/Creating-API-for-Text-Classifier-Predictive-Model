# coding: utf-8
from app import app
import json
import nltk
from sklearn.externals import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from repoze.lru import lru_cache
from app import app
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf

trainingFile = "G:\\textClassifier\\trainingData\\trainingData.json"
trainedModel = "G:\\textClassifier\\trainedModel\\text_classifier.tf"	
trainedDataPKL = "G:\\textClassifier\\trainedModel\\data.pkl"
	
def clean_up_sentence(sentence):
	stemmer = LancasterStemmer()
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	return sentence_words

def bow(sentence, words, show_details=False):
	sentence_words = clean_up_sentence(sentence)
	bag = [0]*len(words)
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print ("found in bag: %s" % w)
	return(np.array(bag))

def classify(sentence):
	model = getModel()
	dsMap = getData()
	words = dsMap['w']
	classes = dsMap['c']
	results = model.predict([bow(sentence, words)])[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append((classes[r[0]], r[1]))
	return return_list

def getModel():
	graph = tf.Graph()
	tf.reset_default_graph()
	with graph.as_default():
		tflearn.config.init_training_mode()
	data = loadXYTrainingSet()
	train_x = data['train_x']
	train_y = data['train_y']
	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 34)
	net = tflearn.fully_connected(net, 34)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)
	print ('accessing tensorboard...')
	myTrainedModel = tflearn.DNN(net,tensorboard_dir='/app/logs/tflearn_logs')
	myTrainedModel.load(trainedModel)
	return myTrainedModel

@lru_cache(maxsize= 10000)
def loadXYTrainingSet():
	return joblib.load(trainedDataPKL)

def getData():
	dsMap = {}
	data = loadXYTrainingSet()
	dsMap['w']= data['words']
	dsMap['c']= data['classes']
	return dsMap

@lru_cache(maxsize=10000)
def getDatasetMap():
	with open(trainingFile) as json_data:
		dataset = json.load(json_data)
	return dataset
	
def response(sentence):
	results = classify(sentence)
	show_details=True
	dataset = getDatasetMap()
	leagueResponse = {}
	if results:
		while results:
			for i in dataset['dataset']:
				founddataset = results[0][0]
				if i['label'] == results[0][0]:
					if show_details: 
						leagueResponse['foundleague']=founddataset
					return leagueResponse
			results.pop(0)	

def extractLeagueClass(text):
	dataset = {}
	league_services = []
	leagueservices = {}
	services = []
	s = response(text)
	if s:
		outcome = s['foundleague']
		leagueservices['leagueCode']= outcome
	return leagueservices
