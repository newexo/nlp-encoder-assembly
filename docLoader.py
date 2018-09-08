import re
from pathlib import Path

def loadDocs(author1, *authors2):
    #load a selection of texts by selected authors
    auths = [author1]
    re1 = re.compile('(\w+)')
    for other_author in authors2:
        a1 = str(other_author)
        match = re1.search(a1)
        if match:
            auths.append(match.group())
    docs = {}
    for author in auths:
        #print(author)
        data_folder = Path("data/" + author)
        idx = 0
        for file in data_folder.iterdir():
            if str(file).endswith(".txt"):
                file_to_open = file
                o = open(file_to_open, 'r')
                documentName = idx
                idx += 1
                document = list(o)
                docs[documentName] = document
    return docs

def trimHeaders(first_document):
    # Determine whether a Project Gutenberg Text
    first_header_index = 0
    second_header_index = 0
    footer_index = 0
    if any("GUTENBERG" in s for s in first_document):
        for first_header_index in range( len(first_document) ):
            if ( ( first_document[first_header_index].find('*END*THE SMALL PRINT!') ) != -1 ) :
                break
            else:
                for first_header_index in range( len(first_document) ):
                    if ( ( first_document[first_header_index].find('START OF THIS PROJECT GUTENBERG') ) != -1 ) :
                        break        
        second_document = list(first_document[first_header_index + 1 :])
        for second_header_index in range( len(second_document) ):
            if ( ( second_document[second_header_index].find('www.gutenberg.org') ) != -1 ) :
                break            
        for footer_index in range( len(first_document) ):
            if ( ( first_document[footer_index].find('End of Project') ) != -1 ) :
                break
            else:
                for footer_index in range( len(first_document) ):
                    if ( ( first_document[footer_index].find('End of the Project') ) != -1 ) :
                        break    
        script = list()
        if (second_header_index < (first_header_index + 100)):
            manuscript = list(first_document[first_header_index +1 + second_header_index +1 : footer_index-1])
        else:
            manuscript = list(first_document[first_header_index +1 : footer_index-1])
    else:
        manuscript = first_document
    return manuscript

def collectLines(script):
    #Compile a list of speakers
    r = re.compile("[A-Z0-9][A-Z0-9]+")
    speakers = []
    for line in script:
        mtch = r.match(line)
        if mtch:
            speakers.append(mtch.group())
    #Omit speakers from the list of text
    s = re.compile(r"\b[A-Z{3}\.]+\b")
    spoken = list(filter(lambda i: not s.search(i), script))
    return speakers, spoken

def sentencer(spoken):
    #Concatenate lines into list entries for future sentence splitting
    newLines = []
    singleLine = ''
    singleLines = []

    #Remove all line returns(ok)
    for j in range(0, len(spoken)):
        spoken[j] = spoken[j].replace('\n', '')

    #Split 5 lines at a time into new list
    for k in range( 0, len(spoken), 3):
        newLines = []
        for line in range( 0, 3 ):
            try:
                newLines.append(' '+spoken[line+k])
            except:
                #print("Index Error at", k, line)
                break
        #Join 5-line groups into one line and append to a list
        singleLine = ''.join(newLines)
        singleLines.append(singleLine)
    
    #Create list of sentences
    sentences = []
    for m in range(0, len(singleLines)):
        mtch = re.findall("[A-Z][^\.!?]*[\.!?]", singleLines[m], re.M|re.I)
        if mtch:
            sentences.append(mtch)
    return sentences

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

    #for t in range(len(phrases)):
        #print(phrases[t])
    
    return phrases
