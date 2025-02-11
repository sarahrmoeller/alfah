#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import string
import csv
import shutil
import copy
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import spacy
from spacy import lookups
from spacy import tokenizer
import nltk
from nltk import wordpunct_tokenize
from nltk import word_tokenize
from nltk import pos_tag
nlp = spacy.load("en_core_web_sm")
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as st
from nltk.stem import WordNetLemmatizer as wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import gensim.models
#import Habituality
import joblib


# In[2]:


import string
import csv
import spacy
from spacy import lookups
from spacy import tokenizer
import re
import nltk
from nltk import wordpunct_tokenize
from nltk import word_tokenize
from nltk import pos_tag
nlp = spacy.load("en_core_web_sm")


# In[3]:


#Imports spreadsheet
path = 'Downloads/new_texts_for_tagging_speaker'
txt_files = glob.glob(path + '/*.txt')
#txt_files = 'Downloads/new_texts_for_tagging_speaker/Speaker Version AAHP 020 Juanita Scott Williams 5-14-2010ufdc_norm.txt'
print(txt_files)
# loop over the list of csv files
for f in txt_files:
    allData = pd.read_csv(f, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
# In[5]:


    # Creates Sentence class
    class Sentence:
        habituality = 0 #using a ternary system: -1 = nonhabitual 0 = unclassified, 1 = habitual
        pos1 = 0; pos2 = 0; pos3 = 0; pos4 = 0; pos5 = 0; pos6 = 0; a1 = 0; a2 = 0; a3 = 0; a4 = 0; a5 = 0;
        synPar1 = 0; synPar2 = 0; synPar3 = 0; synPar4 = 0; r1 = 0; 


    # text is the text of the Sentence
    # num is the index of the line from the original csv
    # be indicates which 'be' is being looked at in the sentence
        # if there are multiple 'be' in a sentence, the first will have be = 1, second will have be = 2, etc to distinguish them
        def __init__(self, text, be, num):
            self.text = text
            self.num = num
            self.be = be


    # In[6]:


    # Imports sentence data and cleans up the text

    lines = []
    with open("new_texts_for_tagging_speaker/Speaker Version AAHP 030 Alvin Butler 06-02-2010ufdc_norm.txt", "r", encoding="utf_8") as file:
    #with open("AAHP 018 Virgil Hayes 3-6-2010ufdc copy.csv", "r", encoding="utf_8") as file:
    #with open("Test Be Sentences.csv", "r", encoding="utf_8") as file:
        lines = file.readlines()

    i = 0
    del lines[0]
    while (i < len(lines)):
        if (lines[i] == "\"\n" or lines[i] == "\"" or lines[i] == ' "\n' or lines[i] == '",,,,,\n'):
            del lines[i]
        index1 = lines[i].index(':')+2
        lines[i] = (lines[i])[index1:-1]
        i +=1
    print(len(lines))


    # In[8]:


    # Duplicates sentences if need be (if they have more than one 'be') but assigns them a value corresponding to their
    # original line index to keep track of the sentence they originated from (to consolidate later)

    sen = []
    for line in range(len(lines)):
            be = 1
            numBe = 0
            parsed = nlp(lines[line])
            for i, word in enumerate(parsed):
                if (word.text.lower() == "be"):
                    numBe += 1
            if (numBe >= 1):
                sen.append(Sentence(lines[line], be, line))
            while (numBe > 1):
                be += 1
                sen.append(Sentence(lines[line], be, line))
                numBe -= 1
    for Sentence in range(len(sen)):
        sen[Sentence].text = (sen[Sentence].text).replace('"','')


    # In[9]:

    if (len(sen) > 0):

        # Creates list of phonetic variation
        phonetic_variation = ["wanna", "tryna", "gonna", "gotta"]  


        # In[10]:


        # Creates arrays representing the parts of speech of the words surrounding 'be'
            # e.g. 'w' is the 'be', 'p' is the previous word, 'a' is the word after, 'pp' is the preprevious word, etc

        w = []
        pppp = []
        ppp = []
        pp = []
        p = []
        a = []
        pa = []
        ppa = []
        pppa = []


        # In[12]:


        # Tags each sentence with rules (detailed rule information in documentation on dropbox)

        index = -1
        senNum = 1
        numBe = 0
        extraBe = []
        beIndices = []
        count = 0

        for Sentence in sen:
            index += 1 
            numBe = 0
            if (index > 0 and (sen[index].text == sen[index-1].text) and (sen[index].num == sen[index-1].num) and (allData.loc[index, 'Speaker'] == allData.loc[index-1, 'Speaker'])):
                senNum += 1
            else:
                senNum = 1
            parsed = nlp(Sentence.text)
            for i, word in enumerate(parsed):
                if (word.text.lower() == "be"):
                    numBe += 1
                    if (numBe == senNum):
                        print(Sentence.text)
                        count +=1
                        w.append(word.pos_)
                        beIndices.append(i)
                        beChildren = list(word.children)
                        if(word.head == word):
                            beSibling = []
                        else:
                            beSibling = list(word.head.children)
                        beDep = word.dep_
                        bePOS = word.pos_
                        beIndex = i
                        preprepreprevious = None
                        prepreprevious = None
                        preprevious = None
                        previous = None
                        after = None
                        postafter = None
                        postpostafter = None
                        postpostpostafter = None
                        if(i >= 4):
                            preprepreprevious = parsed[i-4]
                            pppp.append(preprepreprevious.pos_)
                        else:
                            pppp.append(-1)
                        if(i >= 3):
                            prepreprevious = parsed[i-3]
                            ppp.append(prepreprevious.pos_)
                        else:
                            ppp.append(-1)
                        if(i >= 2):
                            preprevious = parsed[i-2]
                            pp.append(preprevious.pos_)
                        else:
                            pp.append(-1)
                        if(i > 0):
                            previous = parsed[i-1]
                            p.append(previous.pos_)
                        else:
                            p.append(-1)
                        if(i+1 < len(parsed)):
                            after = parsed[i+1]
                            a.append(after.pos_)
                        else:
                            a.append(-1)
                        if(i+2 < len(parsed)):
                            postafter = parsed[i+2]
                            pa.append(postafter.pos_)
                        else:
                            pa.append(-1)
                        if(i+3 < len(parsed)):
                            postpostafter = parsed[i+3]
                            ppa.append(postpostafter.pos_)
                        else:
                            ppa.append(-1)
                        if(i+4 < len(parsed)):
                            postpostpostafter = parsed[i+4]
                            pppa.append(postpostpostafter.pos_)
                        else:
                            pppa.append(-1)

                     #pos1
                        if(previous and (previous.tag_ == "MD" or previous.tag_ == "JJ" or previous.tag_ == "TO")):
                            Sentence.pos1 = 1
                            Sentence.habituality = -1

                    #pos1 pt 2: accounting for phonetic variation
                        if(previous and previous.text in phonetic_variation):
                            Sentence.pos1 = 1
                            Sentence.habituality = -1

                    #pos2
                        if ((after and previous) and (after.tag_ == "JJ" and previous.tag_ != "PRP" and not previous.tag_.__contains__("NN"))):
                            Sentence.pos2 = 1
                            Sentence.habituality = -1

                    #pos3
                        if ((after and previous) and (after.tag_ == "IN" and (previous.tag_ == "VBZ" or previous.tag_ == "VBP"))):
                            Sentence.pos3 = 1
                            Sentence.habituality = -1

                    #pos4
                        if (previous and preprevious) and (previous.tag_ == "NN" and preprevious.tag_ == "JJ"):
                            Sentence.pos4 = 1
                            Sentence.habituality = -1

                    #pos5
                        if (previous and after) and (previous.tag_ == "RB" and (after.tag_ == "PRP" or after.tag_ == "DT")):
                            Sentence.pos5 = 1
                            Sentence.habituality = -1

                    #pos6
                        if (previous and preprevious) and (previous.tag_ == "RB" and preprevious.tag_.__contains__("VB") or preprevious.tag_ == "MD"):
                            Sentence.pos6 = 1
                            Sentence.habituality = -1


                #A1
                        if (((previous and preprevious and prepreprevious) and (previous.text == "n't" and ((preprevious.text == "do") or (preprevious.text == "Do"))) and (prepreprevious.pos_ == "PRON" or prepreprevious.pos_ == "NOUN")) or ((preprevious and prepreprevious and preprepreprevious) and ((previous.pos_ != "VERB" and previous.pos_ != "AUX")) and preprevious.text == "n't" and ((prepreprevious.text == "do") or (prepreprevious.text == "Do")) and (preprepreprevious.pos_ == "PRON" or preprepreprevious.pos_ == "NOUN"))):
                            Sentence.a1 = 1
                            Sentence.habituality = 1

                #A2
                        if (((after) and (after.pos_ == "PUNCT" or after.pos_ == "CCONJ" or after.pos_ == "DET" or after.pos_ == "INTJ" or after.pos_ == "PROPN"))):
                            Sentence.a2 = 1
                            Sentence.habituality = -1


                #A3

                        if ((previous and previous.pos_ == "PRON") or (preprevious and preprevious.pos_ == "PRON" and previous.pos_ != "AUX" and previous.pos_ != "VERB" and previous.pos_ != "PART") or (prepreprevious and prepreprevious.pos_ == "PRON" and previous.pos_ == "ADV" and preprevious.pos_ != "AUX" and preprevious.pos_ != "VERB")):
                            Sentence.a3 = 1
                            Sentence.habituality = 1


                #A4

                        if (after and after.tag_ == "VBG" and previous.pos_ != "AUX" and not (previous.pos_ == "PART" and Sentence.pos1 == 1)):
                            Sentence.a4 = 1
                            Sentence.habituality = 1

                 #A5

                        if ((previous and preprevious) and (previous.text == "n't" and preprevious.text != "do" and preprevious.text != "Do")):
                            Sentence.a5 = 1
                            Sentence.habituality = -1



                  #synPar1 -> aux_children:
                        if len(beChildren) != 0:
                            for child in beChildren:
                                if child.dep_ == "aux" and child.pos_ == "AUX" and child.text.lower() != "do":
                                    Sentence.synPar1 = 1
                                    Sentence.habituality = -1
                    #synPar2 -> aux_siblings
                        if len(beSibling) != 0:
                            for sibling in beSibling:
                                if sibling.dep_ == "aux" and sibling.pos_ == "AUX" and sibling.text.lower() != "do" and sibling.text != "be":
                                    Sentence.synPar2 = 1
                                    Sentence.habituality = -1
                    #synPar3 -> verbal_auxilary
                        if (word.pos_ == "AUX" and word.dep_ == "aux" and word.head.pos_ == "VERB"):
                            Sentence.synPar3 = 1
                            if(Sentence.habituality != -1):
                                Sentence.habituality = 1
                    #synPar4 -> copular_verb
                        if(word.pos_ == "VERB"):
                            Sentence.synPar4 = 1
                            Sentence.habituality = 1

         #R1

                        if ((Sentence.pos1 !=1) and (Sentence.pos2 !=1) and (Sentence.pos3 !=1) and (Sentence.pos4 !=1) and (Sentence.pos5 !=1) and (Sentence.pos6 !=1)):
                            Sentence.r1 = 1
                            Sentence.habituality = 1

                        print(count)


        # In[13]:


        # Uses rule interactions to improve accuracy
            # E.g. If synPar3 and r1 are true, sentence tends to be nonhabitual, therefore set synPar3 is set 
            # to 0 so it becomes a more clear indicator of habituality 
        for Sentence in sen:
            if (Sentence.synPar3 == 1 and Sentence.r1 == 0):
                Sentence.synPar3 = 0
            if (Sentence.synPar3 == 1 and Sentence.a4 == 1):
                Sentence.synPar3 = 0
            if (Sentence.pos5 == 1 and Sentence.a2 == 1 and Sentence.a3 == 1):
                Sentence.pos5 = 0
            if (Sentence.a4 == 1 and Sentence.a5 == 1):
                Sentence.a5 == 0
            if (Sentence.synPar2 == 1 and Sentence.a3 == 1):
                Sentence.synPar2 = 0
            if (Sentence.a3 == 1 and (Sentence.pos6 == 1 or Sentence.synPar1 == 1)):
                Sentence.a3 = 0
            if (Sentence.r1 == 1 and Sentence.synPar1 == 1):
                Sentence.r1 = 0


        # In[14]:


        #Creates spreadsheet document
        with open("rules coraal_analysis_spreadsheet.csv", "w", encoding="utf-8-sig", newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["Habituality", "Sentence", "POS1", "POS2", "POS3", "POS4", "POS5", "POS6","A1", "A2","A3", "A4", "A5", "SynPar1", "SynPar2", "SynPar3", "SynPar4", "R1"])
            for Sentence in sen:
                filewriter.writerow([Sentence.habituality, Sentence.text, Sentence.pos1, Sentence.pos2, Sentence.pos3, Sentence.pos4, Sentence.pos5,Sentence.pos6, Sentence.a1, Sentence.a2, Sentence.a3, Sentence.a4, Sentence.a5, Sentence.synPar1, Sentence.synPar2, Sentence.synPar3, Sentence.synPar4, Sentence.r1])



        # In[15]:


        # Reads in rule spreadsheet
        preData = pd.read_csv("rules coraal_analysis_spreadsheet.csv")


        # In[17]:


        # Creates copy of preData data frame and empties it
        data = copy.deepcopy(preData)
        i = len(preData)-1
        while (i >= 0):
            data.drop(i, inplace=True)
            i = i -1


        # In[18]:


        # Creates 'data' data frame which has a version of each sentence for every 'be' it contains
            # This allows the model to later observe and tag each individual 'be' and will be consolidated at the end

        for Sentence in range(len(sen)):
            numBe = 0
            parsed = nlp(sen[Sentence].text)
            for i, word in enumerate(parsed):
                if (word.text.lower() == "be"):
                    numBe += 1
            if (numBe >= 1):      
                data = pd.concat([data, preData.iloc[[Sentence]]], ignore_index=True, join="inner")
                numBe -=1


        # In[20]:


        # Generating n-grams using the windows of words around 'be'
        # Removing stopwords

        corpus = [] #empty list
        wordnet = wordnet() #object instantiation
        length = len(data['Sentence']) #finding total number of rows
        be_sen = []
        numBe = 1
        for i in data['Sentence']:
            start = 0
            end = 0
            if numBe == 1:
                numBe = 0
                parsed = nlp(i)
                for j, word in enumerate(parsed):
                    if word.text.lower() == "be":
                        numBe += 1
                        if(j >= 4):
                            start = j-4
                        else:
                            start = 0
                        if(len(i) > j+4):
                            end = j+4    
                        else:
                            end = len(i)
                        be_sen.append(parsed[start:end])
            else:
                numBe -= 1


        for i in range(length):
            rev = re.sub('[^a-zA-Z]',' ',str(be_sen[i]))
            rev = rev.lower() #text to lowercase
            rev = rev.split() #each word of the sentence becomes the element of a list
            rev = [wordnet.lemmatize(word) for word in rev if word not in stopwords.words('english')] #lemmatization via list comprehension
            rev = ' '.join(rev) #from list to string
            corpus.append(rev) #appending to the list



        # In[21]:


        # Imports cv and n_gram models to use for prediction
        cv = joblib.load("cv.joblib")
        process = joblib.load("n_gram.joblib")


        # In[22]:


        # Use the cv and n_gram models to predict habituality based solely on n-grams

        x = cv.transform(corpus).toarray()
        predictions = process.predict(x)


        # In[24]:


        # Creates numerical values that match to part of speech tags

        pos_list = pd.DataFrame({'pos': dir(spacy.parts_of_speech)})
        num = []
        for i in range((len(pos_list))):
            num.append(i)
        pos_list.insert(1, 'num', num)
        print(pos_list)


        # In[25]:


        # Assigns part of speech tags to numbers 

        for i in range(len(w)):
            for j in range(len(pos_list)):
                if (w[i] == pos_list.iloc[j, 0]):
                    w[i] = pos_list.iloc[j, 1]
                if (pppp[i] == pos_list.iloc[j, 0]):
                    pppp[i] = pos_list.iloc[j, 1]    
                if (ppp[i] == pos_list.iloc[j, 0]):
                    ppp[i] = pos_list.iloc[j, 1]
                if (pp[i] == pos_list.iloc[j, 0]):
                    pp[i] = pos_list.iloc[j, 1]
                if (p[i] == pos_list.iloc[j, 0]):
                    p[i] = pos_list.iloc[j, 1]
                if (a[i] == pos_list.iloc[j, 0]):
                    a[i] = pos_list.iloc[j, 1]
                if (pa[i] == pos_list.iloc[j, 0]):
                    pa[i] = pos_list.iloc[j, 1]
                if (ppa[i] == pos_list.iloc[j, 0]):
                    ppa[i] = pos_list.iloc[j, 1]
                if (pppa[i] == pos_list.iloc[j, 0]):
                    pppa[i] = pos_list.iloc[j, 1]


        # In[27]:


        # Inserts n-gram and window data into 'data'
        data.insert(len(data.columns), "n-gram", predictions)
        data.insert(len(data.columns), "word", w)
        data.insert(len(data.columns), "preprepreprevious", pppp)
        data.insert(len(data.columns), "prepreprevious", ppp)
        data.insert(len(data.columns), "preprevious", pp)
        data.insert(len(data.columns), "previous", p)
        data.insert(len(data.columns), "after", a)
        data.insert(len(data.columns), "postafter", pa)
        data.insert(len(data.columns), "postpostafter", ppa)
        data.insert(len(data.columns), "postpostpostafter", pppa)


        # In[28]:


        # Creates X data frame which contains the predictor values
        X = data.drop(columns=['Habituality', 'Sentence'], axis=1)


        # In[29]:


        # Load habituality model and predict habituality based on input X
        model = joblib.load("habituality_model.joblib")
        results = model.predict_proba(X)
        results = results[:,0]


        # In[ ]:


        # Sets threshold, where a higher value captures more potential habituals (increased habitual recall)
        for i in range(len(results)):
            if (results[i] >= 0.84):
                results[i] = -1
            else:
                results[i] = 1


        # In[32]:


        # Creates 'temp', a copy of 'data' which tags sentences that have a habitual be with "% HB " in the speaker column
        # In  'data', consolidates the habituality values of each 'be' in a sentence
            # These values are separated by commas, e.g. in "1, -1" the 1 refers to the first be and -1 to the second

        for i in range(len(data)):
            data.loc[i, 'Habituality'] = results[i]
        temp = copy.deepcopy(data)
        temp['Speaker'] = allData['Speaker']
        for i in range(len(temp)):
            if (temp.loc[i, 'Habituality'] == 1):
                   index = temp.loc[i, 'Speaker'].index(':')
                   temp.loc[i, 'Speaker'] = temp.loc[i, 'Speaker'][:index+1] + " %HB "
        for i in range(len(temp)):
                    index = i+1
                    while (index < len(temp)) and (temp.loc[index, 'Sentence'].strip(",\n\t ") == temp.loc[i, 'Sentence'].strip(",\n\t ")):
                        if (temp.loc[index, 'Habituality'] == 1):
                            temp.loc[i, 'Speaker'] = temp.loc[index, 'Speaker']
                        index+=1
        for i in range(len(data)-1):
            j = i+1 
            while (data.loc[i, 'Sentence'] == data.loc[j, 'Sentence']):
                data.loc[i, 'Habituality'] = ("" + str(data.loc[i, 'Habituality']) + "," + str(data.loc[j, 'Habituality']))
                if (j < (len(data)-1)):
                    j += 1
                else:
                    break


        # In[33]:


        # Prints data to file for next step

        with open('temp.csv', 'w+', newline='') as file:
           temp.to_csv('temp.csv')


        # In[36]:


        # Returns to allData (original dataset) and assigns it the habituality values from 'data' and the sentence and 
        # speaker values from 'temp'
        # Finally if the sentence has no 'be' (and therefore was not yet tagged), the habituality tag is set to '0'

        noBe = True
        former = ''
        for i in range(len(allData)):
            noBe = True
            for j in range(len(data)):
                if (data.loc[j, 'Sentence'].strip(",\n ") != former.strip(",\n ")):
                    former = data.loc[j, 'Sentence']
                    if allData.loc[i, 'Sentence'].strip(",\n ") == data.loc[j, 'Sentence'].strip(",\n\t "):
                        allData.loc[i, 'Habituality'] = data.loc[j, 'Habituality']
                        allData.loc[i, 'Sentence'] = temp.loc[j, 'Sentence'].strip(",\n ")
                        allData.loc[i, 'Speaker'] = temp.loc[j, 'Speaker']
                        noBe = False
            if noBe == True:
                allData.loc[i, 'Habituality'] = 0   

        index = 0

    else:
            for i in range(len(allData)):
                allData.loc[i, 'habitualBe']  = 0

    index = f.index("/") + 1
    f = f[index:]
    filename = 'Downloads/Habitual Be Annotated' + "/" + "Annotated " + f 
    print(filename)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w+', newline='') as file:
        allData.to_csv(filename, sep = '\t')
