# coding: utf-8
from app import app
import json
from flask import Flask, jsonify, request
from app.utility import textClassPredictor as tcp
import sys
from importlib import reload
reload(sys)

processedCodes = {}

@app.route('/api/v1.0/leagues', methods=['POST'])
def getServices():
	requestData = json.loads(request.data)
	text = requestData['notes']
	k = text.replace(" ","")
	if processedCodes and k in processedCodes:
		out = processedCodes[k]
	else:
		out = tcp.extractLeagueClass(text)
		processedCodes[k] = out
	return json.dumps(out,sort_keys=True, indent=4)

