import re
from pathlib import Path

def readStopList():
    #Clean the stopword list
    stoplist = []
    clean_line = []
    data_folder = Path("data/")
    file_to_open = data_folder / "snowball_stop.txt"
    f = open(file_to_open, 'r')
    full_stop = list(f)

    for n in range( 0, len(full_stop), 1 ):
        clean_line = full_stop[n].split('|')
        clean_line[0] = clean_line[0].replace(' ', '')
        stoplist.append(clean_line[0])

    for p in range(len(stoplist)):
        stoplist[p] = stoplist[p].replace('\n', '')

    #print(stoplist)
    return stoplist

def collectPhrases(sentences, stoplist):
    # Create list of phrases using stopwords
    phrases = []
    candidate_phrases = []

    for q in range(len(sentences)):
        for r in sentences[q]:
            words = re.split("\\s+", r)
            previous_stop = False
     
            # Examine each word to determine if it is a phrase boundary marker or part of a phrase or alone
            for w in words:
     
                if w in stoplist and not previous_stop:
                    # phrase boundary encountered, so put a hard indicator
                    candidate_phrases.append(";")
                    previous_stop = True
                elif w not in stoplist and len(w) > 3:
                    # keep adding words to list until a phrase boundary is detected
                    candidate_phrases.append(w.strip())
                    previous_stop = False
     
        # Create a list of candidate phrases without boundary demarcation
        phrases = re.split(";+", ' '.join(candidate_phrases))

    # Clean up phrases    
    re2 = re.compile('[^\.!?,"(){}\*:]*[\.!?,"(){}\*:]')
    for s in range(len(phrases)):
        phrases[s] = re.sub(re2, '', phrases[s])
        phrases[s] = phrases[s].strip(' ')
        phrases[s] = phrases[s].replace(' ', '_')
        phrases[s] = phrases[s].replace('__', '_')
        phrases[s] = phrases[s].strip('_')

    for s in range(len(phrases)):
        try:
            phrases.remove('')
            phrases.remove(' ')
            phrases.remove('/n')
        except:
            pass
    
    return phrases


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