# coding: utf-8
from app import app
import json
from flask import Flask, jsonify, request
from app.utility import textClassificationModeler as tcm
import sys
import logging
from importlib import reload
reload(sys)



@app.route('/api/v1.0/train/textClassifier/model', methods=['GET'])
def textClassifierTrainer():
	tcm.textClassifierModeler()
	return json.dumps({"response":"Training Completed"})