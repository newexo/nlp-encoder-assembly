import os, sys
import re
import json
from triggers import *
from pathlib import Path

#Example path_string : "data/mudlogs/"
def pull_logs(path_string):
	data_folder = Path(path_string)
	idx = 0
	doc = {}
	document = []
	for file in data_folder.iterdir():
	    if str(file).endswith("log"):
	        file_to_open = file
	        o = open(file_to_open, 'r')
	        documentName = idx
	        idx += 1
	        print(idx)
	        document = list(o)
	        doc[documentName] = document
	return document

#Expects a list of log lines
def filter_desc(log_doc):
	buff = [json.loads(line) for line in document[:]]
	description = []
	index = -1
	indices = []
	descripDict = {}
    descr = [d for d in tenk if d['tag'] == 'mudline']
    for l in descr[:]:
	    li = l['line']
	    ansi_rem = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~](\\?\;?\d+)*m')
	    p = ansi_rem.sub('', str(li))
	    print(p) #for nice output
		description.append(a)
		index += 1
		indices.append(index)
		descripDict = dict(zip(indices, description))

	#print(index) #for number of lines
	return descripDict


def filter_actions(log_doc):
	buff = [json.loads(line) for line in document[:]]
	action = []
	index = -1
	indices = []
	actionDict = {}
	act = [d for d in tenk if d['tag'] == 'playerline']
	for l in act[:]:
	    li = l['line']
	    ansi_rem = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~](\\?\;?\d+)*m')
	    p = ansi_rem.sub('', str(li))
	    print(p)
		action.append(a)
		index += 1
		indices.append(index)
		actionDict = dict(zip(indices, action))	    
	#print(index) #for number of lines
	return actionDict	