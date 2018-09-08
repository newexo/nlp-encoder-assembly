def collectWords(sentences):
    #Establish wordList
    wordList = []
    for u in range(len(sentences)):
        for v in sentences[u]:
            words = re.split("\\s+", v)
            wordList.extend(words)
    #Establish wordDict
    wordDict = {}
    for w in range(len(wordList)):
        newWord = wordList[w]
        newWord = newWord.lower()
        newWord = newWord.replace('.', '')
        wordDict[w] = newWord
    return wordDict

def wordFreq(wordDict):
    #Perform word counts on dict
    countDict = {}
    for x in range(len(wordDict)):
        term = wordDict[x]
        #print(wordDict)
        count = 1
        for y in range(len(wordDict)):
            try:
                if wordDict[y].find(term) > 0:
                    count += 1
            except:
                pass
            countDict[term] = count #MAJOR ERROR HERE "TypeError: unhashable type: 'dict'"

    for k, v in countDict.items():
        print(k, v)
    return countDict

def computeIDF(docList):
    # Calculates the weight of rare words across all docs
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

