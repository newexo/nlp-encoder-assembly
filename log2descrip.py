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
	description = []
	index = -1
	indices = []
	descripDict = {}
	for line in log_doc: #in the end, append all good lines to a list then dict
	    ansi_rem = re.split(r'(37m)|(33m)', line, 100)
	    stripFront = ansi_rem
	    l = len(ansi_rem)
	    #print(stripFront[l-1])
	    ansi_rem2 = re.split(r'(\\)', stripFront[l-1], 100)
	    stripBack = ansi_rem2
	    a = stripBack[0]
	    l = len(a)
	    if (a.startswith('{"mud": "sloth", "line": "') ):
	        a = a[26:(l-74)]
	    if (a.startswith('", "tag": "mudline",') ):
	        a = ''
	    if (a.startswith('",') ):
	        a = ''
	    if (a.startswith('"}') ):
	        a = ''
	    if (a.startswith('***') ):
	        a = '' 
	    if ('timestamp' in a):
	        a = ''
	    if ('http' in a):
	        a = ''
	    if (a != ''):
	        #print(a) #for nice output
	        description.append(a)
	    index += 1
	    indices.append(index)
	descripDict = dict(zip(indices, description))

	#print(index) #for number of lines
	return descripDict