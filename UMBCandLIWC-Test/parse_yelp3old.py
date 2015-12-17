from __future__ import division

import os, timeit, nltk, pickle, warnings, sys
warnings.filterwarnings("ignore")
import json, codecs
from collections import defaultdict, Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy
from requests import get
from gensim.models import word2vec
#from gensim.models import doc2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import wordnet as wn
#import pattern.en
from operator import itemgetter
import sklearn
from sklearn import linear_model

FREVIEW = os.path.join('..', 'data', 'yelp_academic_dataset_review.json')
FBUSINESS = os.path.join('..', 'data', 'yelp_academic_dataset_business.json')

def getReviews(quantOfRest=100000, quantOfReviewsPerRest=5000):
    restaurantIDs = []
    with codecs.open(FBUSINESS,'rU','utf-8') as f:
        for business in f:
            if "Restaurants" in json.loads(business)["categories"]:
                restaurantIDs.append(json.loads(business)['business_id'])
    ########
    dictRestaurantIDsToReview = {}
    with codecs.open(FREVIEW,'rU','utf-8') as f:
        for review in f:
            reviewText = json.loads(review)['text']
            ID = json.loads(review)['business_id']
            if ID in restaurantIDs:
                if ID in dictRestaurantIDsToReview.keys():
                    if len(dictRestaurantIDsToReview.get(ID)) < quantOfReviewsPerRest:
                        dictRestaurantIDsToReview.get(ID).append(reviewText)
                else:
                    if len(dictRestaurantIDsToReview.keys()) < quantOfRest:
                        dictRestaurantIDsToReview[ID] = [reviewText]
                    else:
                        break
    return dictRestaurantIDsToReview

def getRestIDFoodItemSentencelist(restaurantID, review, posSeeds, negSeeds):
    posSeedsRestaurantIDFoodItemSentencelist = []
    posFound = False
    negSeedsRestaurantIDFoodItemSentencelist = []
    negFound = False
    restaurantIDFoodItemSentencelist = [] #[[restaurantid, foodItem, sentence]...]
    sents = nltk.sent_tokenize(review)
    for sidx in range(len(sents)):
        reviewFoodItems = []
        reconstructedSent = []
        sent = sents[sidx]
        sent = ' '.join(sent.split()) # get rid of extra white spaces
        sent = sent.strip() # necessary? 
        if sent in posSeeds:
            posFound = True
        elif sent in negSeeds:
            negFound = True
        words = nltk.word_tokenize(sent)
        #words_pos = nltk.pos_tag(words)
        if not sent: continue
        originalSent = sent
        for i, word in enumerate(words):
            word = word.strip()
            word = word.lower()
            #if word == "not":
            #    print "this could be a good seed to add which i've done yet: ", sent
            """if word == "hate":
                print "this could be a good seed to add which i've done yet: ", sent
            if word == "disgusting":
                print "this could be a good seed to add which i've done yet: ", sent
            if word == "too":
                print "this could be a good seed to add which i've done yet: ", sent
            if word == "don't":
                print "this could be a good seed to add which i've done yet: ", sent"""
            #pos = words_pos[i][1]
            if not word: continue
            synsets = wn.synsets(word)
            #if pos == "NNS" or pos == "NNPS" or pos == 'NN' or pos == 'NNP':
                #word = pattern.en.singularize(word)
            found = False
            for synset in synsets:
                if synset._lexname == "noun.food":
                    found = True
                    #if word in extFoodItems:
                    #word = pattern.en.singularize(word)
                    reviewFoodItems.append(word)
                    reconstructedSent.append("food")
                    #reconstructedSent.append(word)
                    break
            if not found:
                reconstructedSent.append(word)
        for reviewFoodItem in reviewFoodItems:
            restaurantIDFoodItemSentencelist.append([restaurantID, reviewFoodItem, ' '.join(reconstructedSent), originalSent])
        if posFound:
            posSeedsRestaurantIDFoodItemSentencelist.append([restaurantID, reviewFoodItem, ' '.join(reconstructedSent), originalSent])
            posFound = False
        if negFound:
            negSeedsRestaurantIDFoodItemSentencelist.append([restaurantID, reviewFoodItem, ' '.join(reconstructedSent), originalSent])
            negFound = False
    return restaurantIDFoodItemSentencelist, posSeedsRestaurantIDFoodItemSentencelist, negSeedsRestaurantIDFoodItemSentencelist

def getRestaurantIDFoodItemSentencelist(dictRestaurantIDsToReview, posSeeds, negSeeds):
    posSeedsRestaurantIDFoodItemSentencelist = []
    negSeedsRestaurantIDFoodItemSentencelist = []
    restaurantIDFoodItemSentencelist = [] #[[restaurantid, foodItem, sentence]...]
    nofood = 0
    food = 0
    for restaurantID in dictRestaurantIDsToReview.keys():
        #print "doing restaurantID: ", restaurantID
        count = 0
        for review in dictRestaurantIDsToReview.get(restaurantID):
            count += 1
            addOn, posSeedAddOn, negSeedAddon = getRestIDFoodItemSentencelist(restaurantID, review, posSeeds, negSeeds)
            restaurantIDFoodItemSentencelist += addOn
            posSeedsRestaurantIDFoodItemSentencelist += posSeedAddOn
            negSeedsRestaurantIDFoodItemSentencelist += negSeedAddon
            if addOn:
                #print "  Found a food item in review", count
                #for ff in addOn:
                #    print "   ", ff[1], ":  ", ff[2] 
                food += 1
                #print ""
            else:
                #print "  NO FOOD ITEM FOUND! review", count
                #print "   ", "review:", review
                nofood += 1
                #print "" 
    print "total restaurant reviews that don't mention food: ", nofood
    print "total restaurant reviews that do mention food: ", food
    return restaurantIDFoodItemSentencelist, posSeedsRestaurantIDFoodItemSentencelist, negSeedsRestaurantIDFoodItemSentencelist

def tfidfTransform(sentences):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_counts = count_vect.fit_transform(sentences)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    X_tfidf_normalized = sklearn.preprocessing.normalize(X_tfidf, norm='l1', axis=1)
    return X_tfidf_normalized, count_vect.vocabulary_

def getWordVec(word, w2vModel, uniqueWordsThatDidNotWork):
    word = word.lower()
    if word in w2vModel.vocab:
        vec = w2vModel.syn0norm[w2vModel.vocab[word].index]
    else:
        if not (word in uniqueWordsThatDidNotWork): 
            #uniqueWordsThatDidNotWork.append(word)
            #print "getWordVec(", word, ") did not work."
            pass
        raise KeyError("word '%s' not in vocabulary" % word)
    return vec     

def getSentVec(sentence, sentenceIndexVal, wordWeights, vocab, w2vModel, uniqueWordsThatDidNotWork):
    reviewSentVec = np.array([0.0]*300)
    sentTokenized = nltk.word_tokenize(sentence)
    wordCount = 0
    for word in sentTokenized:
        try: 
            wordW2vVec = getWordVec(word, w2vModel, uniqueWordsThatDidNotWork)
        except:
            if not (word in uniqueWordsThatDidNotWork): 
                uniqueWordsThatDidNotWork.append(word)
                #print "getSentVec(", word, ") did not work.  Not including", word, "in", sentence
            continue
        try: 
            sentenceIndexVal = int(sentenceIndexVal) ; vocabIndexVal = int(vocab.get(word))
        except: 
            if not (word in uniqueWordsThatDidNotWork): 
                uniqueWordsThatDidNotWork.append(word)
                #print "sentenceIndexVal or vocabIndexVal was not valid:", sentenceIndexVal, vocab.get(word), "Not including", word, "in", sentence
            continue
        #print "found ", word
        wordCount += 1
        wordWeight = wordWeights[sentenceIndexVal, vocabIndexVal]
        reviewSentVec += wordW2vVec*wordWeight
    return reviewSentVec

def createVecsFromSentences(restIDsFoodItemsSents, wordWeights, vocab, sentences, w2vModel, uniqueWordsThatDidNotWork):
    # dict restID to [[foodItem, foodItemVec, sent, sentVec]...]
    dictRestIDToFoodItemFoodItemVecSentSentVec = {}
    for i, restIDsFoodItemsSent in enumerate(restIDsFoodItemsSents):
        if (i % 1567) == 0: sys.stdout.write('.')
        restaurantID, foodItem, sentence, originalSent = restIDsFoodItemsSent
        try: foodItemVec = getWordVec(foodItem, w2vModel, uniqueWordsThatDidNotWork)
        except: continue
        sentVec = getSentVec(sentence, sentences.index(sentence), wordWeights, vocab, w2vModel, uniqueWordsThatDidNotWork)
        if dictRestIDToFoodItemFoodItemVecSentSentVec.has_key(restaurantID): #restaurant in dict already with at least one [foodItem, foodItemVec, sentence, sentVec]
            dictRestIDToFoodItemFoodItemVecSentSentVec[restaurantID].append([foodItem, foodItemVec, sentence, sentVec, originalSent])
        else: #new restaurant
            dictRestIDToFoodItemFoodItemVecSentSentVec[restaurantID] = [[foodItem, foodItemVec, sentence, sentVec, originalSent]]
    return dictRestIDToFoodItemFoodItemVecSentSentVec

def convertFiveListsToDict(fiveLists):
    newDict = {}
    for fiveList in fiveLists:
        if newDict.has_key(fiveList[0]):
            newDict.get(fiveList[0]).append(fiveList[1:])
        else:
            newDict[fiveList[0]] = [fiveList[1:]]
    return newDict

def removeSeedsFromDict(restDict, posNegSeeds):
    posnegDict = convertFiveListsToDict(posNegSeeds)
    #change dictRestIDToFoodItemFoodItemVecSentSentVec to be the subtraction of dictPosRestIDToFoodItemFoodItemVecSentSentVec and dictNeg...
    restSet = set(restDict.keys())
    posnegSet = set(posnegDict.keys())
    intersectKeys = restSet.intersection(posnegSet)
    for key in intersectKeys:
        for i, restlist in enumerate(restDict.get(key)):
            for j, posNeglist in enumerate(posnegDict.get(key)):
                if (restlist[0] == posNeglist[0]) and (restlist[2] == posNeglist[2]):
                    try:
                        restDict.get(key).pop(i)
                    except:
                        print "error:", len(restDict.get(key))
                    #after sucessful removal, check to see if there's anything left for that restaurant
                    if restDict.get(key) == []:
                        restDict.pop(key)
                    break

def getReviewSentencesNNs(dictRestIDToFoodItemFoodItemVecSentSentVec, posNegSeeds, withinDis=.25): 
    restIDfoodItemFoodItemVecSentSentVecs = makeDictAFiveLists(dictRestIDToFoodItemFoodItemVecSentSentVec)
    X = map(itemgetter(4), restIDfoodItemFoodItemVecSentSentVecs)
    Y = map(itemgetter(4), posNegSeeds)  #posNegSeed[4]
    NNs = []
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #Y = np.array([[0,0]])
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(X)
    try:
        allDistances, indices = nbrs.kneighbors(Y)
    except:
        print "allDistances", allDistances
    for i, distancesFromASeed in enumerate(allDistances):
        for j, dist in enumerate(distancesFromASeed): #0 since Y is just one seed we are looking at at a time
            if dist <= withinDis:
                if (dist == 0.0) and ( posNegSeeds[i][1] in map(itemgetter(0), dictRestIDToFoodItemFoodItemVecSentSentVec.get(restIDfoodItemFoodItemVecSentSentVecs[indices[i,j]][0])) ) and ( posNegSeeds[i][3] in map(itemgetter(2), dictRestIDToFoodItemFoodItemVecSentSentVec.get(restIDfoodItemFoodItemVecSentSentVecs[indices[i,j]][0])) ):
                    #and (restIDfoodItemFoodItemVecSentSentVecs[indices[i,j]][1] == posNegSeeds[i][1]) and (restIDfoodItemFoodItemVecSentSentVecs[indices[i,j]][3] == posNegSeeds[i][3]):
                    beforeCount = dictCount(dictRestIDToFoodItemFoodItemVecSentSentVec)
                    removeSeedsFromDict(dictRestIDToFoodItemFoodItemVecSentSentVec, [posNegSeeds[i]]) #perhaps this happens where it didn't find it earlier, i'm not sure still figuring it out
                    if dictCount(dictRestIDToFoodItemFoodItemVecSentSentVec) == beforeCount:
                        #print("try to remove a seed during getReviewSentencesNNs: no sucess")
                        removeSeedsFromDict(dictRestIDToFoodItemFoodItemVecSentSentVec, [posNegSeeds[i]])
                    else:
                        pass
                        #print("try to remove a seed during getReviewSentencesNNs: sucess")
                else:
                    NNs.append(restIDfoodItemFoodItemVecSentSentVecs[indices[i,j]])
            else:
                break #the rest of the distances will only get further away
    return NNs

def LRTest(NNs, clf, posThreshold=.6, negThreshold=.6): #restID foodItem FoodItemVec Sent SentVecs
    #predict = clf.predict_proba(NN[4])
    predict = map(itemgetter(4), NNs)
    if clf.classes_[0] == 0:
        negProbability = predict[0][0]
        posProbability = predict[0][1]
    elif clf.classes_[0] == 1:
        negProbability = predict[0][1]
        posProbability = predict[0][0]
    #rint "LR",negProbability, posProbability
    if negProbability >= negThreshold:
        return 0
    if posProbability >= posThreshold:
        return 1
    else:
        return -1

def addFiveListToDict(myDict, fiveList):
    if myDict.has_key(fiveList[0]):
        foodItems = map(itemgetter(0), myDict.get(fiveList[0]))
        sents = map(itemgetter(2), myDict.get(fiveList[0]))
        if not ((fiveList[1] in foodItems) and (fiveList[3] in sents)):
            myDict.get(fiveList[0]).append(fiveList[1:])
    else:
        myDict[fiveList[0]] = [fiveList[1:]]

def makeDictAFiveLists(myDict):
    fiveLists = []
    for key in myDict.keys():
        for fourList in myDict.get(key):
            fiveLists.append([key] + fourList[:])
    return fiveLists

def dictCount(myDict):
    total = 0
    for key in myDict.keys():
        total += len(myDict.get(key))
    return total

def addSentsThatTalkAboutThatFoodItemInRestAutoIntoSeeds(dictRestIDToFoodItemFoodItemVecSentSentVec, newPosSeeds, newNegSeeds, NNs):
    toAddPos = []
    toAddNeg = []
    for newPosSeed in newPosSeeds: #[[restID foodItem FoodItemVec Sent SentVecs OriginalSent]..]
        if dictRestIDToFoodItemFoodItemVecSentSentVec.has_key(newPosSeed[0]):
            for foodItemFoodItemVecSentSentVec in dictRestIDToFoodItemFoodItemVecSentSentVec.get(newPosSeed[0]):
                if (newPosSeed[1] == foodItemFoodItemVecSentSentVec[0]) and (newPosSeed[3] != foodItemFoodItemVecSentSentVec[2]):
                    toAddPos.append( [newPosSeed[0]] + foodItemFoodItemVecSentSentVec )
    NNs += toAddPos
    for newNegSeed in newNegSeeds: #[[restID foodItem FoodItemVec Sent SentVecs OriginalSent]..]
        if dictRestIDToFoodItemFoodItemVecSentSentVec.has_key(newNegSeed[0]):
            for foodItemFoodItemVecSentSentVec in dictRestIDToFoodItemFoodItemVecSentSentVec.get(newNegSeed[0]):
                if (newNegSeed[1] == foodItemFoodItemVecSentSentVec[0]) and (newNegSeed[3] != foodItemFoodItemVecSentSentVec[2]):
                    toAddNeg.append( [newNegSeed[0]] + foodItemFoodItemVecSentSentVec )
    NNs += toAddNeg

def removeDuplicates(myDict):
    for key in myDict.keys():
        s = []
        for i in myDict.get(key):
            notIn = True
            for j in s:
                if (i[0] == j[0]) and (i[2] == j[2]):
                    notIn = False
                    break
            if notIn:
                s.append(i)
        myDict[key] = s
   
def performOfflineSteps():
    #OFFLINE TIME...
    print "Part 1 (extraction of dataset)...\n"
    posSeeds = ["The fish was very good, but the Reuben was to die for!", "That buffalo chicken pizza was awesome.", "They serve waffles in halves, which is great.", "The eggs and toast are good, the homemade hot sausage is excellent.", "My recommendation is a breakfast sandwich on a bagel and a coffee.", "You MUST try their burgers.", "Another must are the fries with Guinness gravy and if you haven't had enough go for the ultimate sleeper, Leek and potato soup.", "Oddly enough I think my favorite thing to eat there is the wonton soup, never had better.", "I had chicken picatta...it was perfect.", "The shakes are good, as are the shoestring fries and baked beans."]
    negSeeds = ["I got about half-way through eating the dish and the so-so aioli became too much for me.", "The side salad was a bit lacking-- it's just iceberg lettuce (my mom says there seemed to be romaine too) with some olives and a couple grape tomatoes.","And daal makhini is similarly too saucy and not in a yummy sauce way.", "Too much cheese on their pizzas, and the customer service is incredibly poor.","The drawback are the barely cooked potatoes.", "I found many of the beers to be quite tasty, however, nothing exactly special about them.", "I generally enjoy steamed dumplings but these dumplings(too much dumpling little meat) werent so good.", "The loaded potatoes were more like rubber than potatoes and not what I was expecting.", "I got a cheese pizza delivery from here once, and I feel like...it's one of those places where it's okay and if that's what someone put in front of me I would eat it, but I definitely wouldn't go, OMG, let's eat there!", "One of my friends ordered a Buffalo Chicken Sandwich, which according to the menu description, should've had lettuce and tomato in it. But the sandwich that arrived on our table had none of those ingredients.", "Don't get the salted pork chops - save that craving for Orient Kitchen.", "And since the pizza is just mediocre it is definitely not a good value so I'll pass!", "The chicken soup was horrible.", "Fried rice were ok but I've had much much better at other places."] 
    #do one pass to get [[restaurantid, foodItem, sentence]...]
    dictRestaurantIDsToReview = getReviews(quantOfRest=200, quantOfReviewsPerRest=400)
    restaurantIDFoodItemSentencelist, posSeedsRestaurantIDFoodItemSentencelist, negSeedsRestaurantIDFoodItemSentencelist = getRestaurantIDFoodItemSentencelist(dictRestaurantIDsToReview, posSeeds, negSeeds)
    pickle.dump(restaurantIDFoodItemSentencelist, open("restaurantIDFoodItemSentencelist.p", "w+"))    
    pickle.dump(posSeedsRestaurantIDFoodItemSentencelist, open("posSeedsRestaurantIDFoodItemSentencelist.p", "w+"))
    pickle.dump(negSeedsRestaurantIDFoodItemSentencelist, open("negSeedsRestaurantIDFoodItemSentencelist.p", "w+"))
        
    print "\nPart 2 (calculate vectors for everything and create dicts for all reviews, pos/neg reviews)...\n"
    restIDsFoodItemsSents = pickle.load(open("restaurantIDFoodItemSentencelist.p", "r"))
    posSeedsRestaurantIDFoodItemSentencelist = pickle.load(open("posSeedsRestaurantIDFoodItemSentencelist.p", "r"))
    negSeedsRestaurantIDFoodItemSentencelist = pickle.load(open("negSeedsRestaurantIDFoodItemSentencelist.p", "r"))
    #vectorization
    sentences = map(itemgetter(2), restIDsFoodItemsSents)
    wordWeights, vocab = tfidfTransform(sentences)  
    w2vModel = word2vec.Word2Vec.load_word2vec_format(os.path.join('..', 'word2vec-Test', 'trunk', 'GoogleNews-vectors-negative300.bin'), binary=True)  #
    uniqueWordsThatDidNotWork = []
    dictRestIDToFoodItemFoodItemVecSentSentVec = createVecsFromSentences(restIDsFoodItemsSents, wordWeights, vocab, sentences, w2vModel, uniqueWordsThatDidNotWork)
    removeDuplicates(dictRestIDToFoodItemFoodItemVecSentSentVec)
    dictPosRestIDToFoodItemFoodItemVecSentSentVec = createVecsFromSentences(posSeedsRestaurantIDFoodItemSentencelist, wordWeights, vocab, sentences, w2vModel, uniqueWordsThatDidNotWork)
    dictNegRestIDToFoodItemFoodItemVecSentSentVec = createVecsFromSentences(negSeedsRestaurantIDFoodItemSentencelist, wordWeights, vocab, sentences, w2vModel, uniqueWordsThatDidNotWork)
    pickle.dump(dictRestIDToFoodItemFoodItemVecSentSentVec, open("dictRestIDToFoodItemFoodItemVecSentSentVec.p", "w+"))    
    pickle.dump(dictPosRestIDToFoodItemFoodItemVecSentSentVec, open("dictPosRestIDToFoodItemFoodItemVecSentSentVec.p", "w+"))
    pickle.dump(dictNegRestIDToFoodItemFoodItemVecSentSentVec, open("dictNegRestIDToFoodItemFoodItemVecSentSentVec.p", "w+"))
    print ""
    print "total review sentences that has foodItems to be looked at:", dictCount(dictRestIDToFoodItemFoodItemVecSentSentVec), "(duplicates exist when more than one food item is found in a given sentence)"
    print "word2vec database could not find ", len(uniqueWordsThatDidNotWork), "unique words.  Here are some of them: ", uniqueWordsThatDidNotWork[:10]

if __name__ == "__main__":
    #performOfflineSteps()
    
    #ONLIME TIME...
    print "\nPart 3 (Bootstrap)...\n"
    dictRestIDToFoodItemFoodItemVecSentSentVec = pickle.load(open("dictRestIDToFoodItemFoodItemVecSentSentVec.p", "r")) #dictionary of all review sentences
    dictPosRestIDToFoodItemFoodItemVecSentSentVec = pickle.load(open("dictPosRestIDToFoodItemFoodItemVecSentSentVec.p", "r")) #positive seeds (in the form of a dictionary. Later I go back and forth from dictionary to list of lists
    dictNegRestIDToFoodItemFoodItemVecSentSentVec = pickle.load(open("dictNegRestIDToFoodItemFoodItemVecSentSentVec.p", "r")) #negative seeds
    #create LR clf
    logistic = linear_model.LogisticRegression()
    #duplicate original pos/neg seeds to be new pos/neg seeds.  
    newPosSeeds = makeDictAFiveLists(dictPosRestIDToFoodItemFoodItemVecSentSentVec)
    newNegSeeds = makeDictAFiveLists(dictNegRestIDToFoodItemFoodItemVecSentSentVec)
    print "  initial Pos Seeds added:", len(newPosSeeds)
    print "  initial Neg Seeds added:", len(newNegSeeds)
    print ""
    #bootstrap
    i = 1
    while(i < 20):
        if (i % 5) == 0: #remove duplicates every once in a while because there's an error removing seeds which may cause duplicates to crop up and i can't find the problem.
            removeDuplicates(dictRestIDToFoodItemFoodItemVecSentSentVec)
            removeDuplicates(dictPosRestIDToFoodItemFoodItemVecSentSentVec)
            removeDuplicates(dictNegRestIDToFoodItemFoodItemVecSentSentVec)
        #train on new pos/neg seeds
        trainPosSeeds = makeDictAFiveLists(dictPosRestIDToFoodItemFoodItemVecSentSentVec)
        trainNegSeeds = makeDictAFiveLists(dictNegRestIDToFoodItemFoodItemVecSentSentVec)
        clf = logistic.fit( map(itemgetter(4), trainPosSeeds) + map(itemgetter(4), trainNegSeeds), [1]*len(trainPosSeeds)+[0]*len(trainNegSeeds)) #could be faster not training on so many
        #remove new seeds from the dictionary of all review sentences
        beforeReviewSentTotal = dictCount(dictRestIDToFoodItemFoodItemVecSentSentVec)
        removeSeedsFromDict(dictRestIDToFoodItemFoodItemVecSentSentVec, makeDictAFiveLists(dictPosRestIDToFoodItemFoodItemVecSentSentVec))
        removeSeedsFromDict(dictRestIDToFoodItemFoodItemVecSentSentVec, makeDictAFiveLists(dictNegRestIDToFoodItemFoodItemVecSentSentVec))
        print "iteration", i 
        print "  from previous iteration removed", beforeReviewSentTotal-dictCount(dictRestIDToFoodItemFoodItemVecSentSentVec), "sentences from all reviews"
        #get NNs of new pos/neg seeds --kind of like knn compare get-all-sentVec with each-new-seed-sentVec )
        NNs = getReviewSentencesNNs(dictRestIDToFoodItemFoodItemVecSentSentVec, newPosSeeds+newNegSeeds)
        w2vNNsCount = len(NNs)
        print "  NN candidates gotten from word2vec:", w2vNNsCount
        #add sentences where the same foodItems are being discussed in a given restaurant to NNs
        addSentsThatTalkAboutThatFoodItemInRestAutoIntoSeeds(dictRestIDToFoodItemFoodItemVecSentSentVec, newPosSeeds, newNegSeeds, NNs)
        print "  NN candidates gotten from sents that share same food item:", len(NNs) - w2vNNsCount
        #set new pos/neg seeds = []
        newPosSeeds = []
        newNegSeeds = []
        #test NNs in LR clf, add the ones that pass to both original pos/neg seeds and to new pos/neg seeds
        posCutoff = 0
        negCutoff = 0
        for NN in NNs: #could be faster through batch classification 
            if (posCutoff > 200) and (negCutoff > 200):
                break 
            posThreshold=.60+i*.01
            negThreshold=.52+i*.04
            if posThreshold > .65:
                posThreshold = .65
            if negThreshold > .62:
                negThreshold = .62
            if len(NNs) > 500:
                posThreshold = .8
                negThreshold = .8
            if len(NNs) > 1000:
                posThreshold = .9
                negThreshold = .9
            label = LRTest(NN, clf, posThreshold=posThreshold, negThreshold=negThreshold)
            if label==1: #positive
                if posCutoff > 200:
                    continue
                posCutoff += 1
                addFiveListToDict(dictPosRestIDToFoodItemFoodItemVecSentSentVec, NN)
                newPosSeeds.append(NN)
            elif label==0: #negative
                if negCutoff > 200:
                    continue
                negCutoff += 1
                addFiveListToDict(dictNegRestIDToFoodItemFoodItemVecSentSentVec, NN)
                newNegSeeds.append(NN)
        if (not newPosSeeds) and (not newNegSeeds): #no new seeds to work with, so you're done
            if not ((posCutoff > 200) or (negCutoff > 200)):
                break
        print "  total newPosSeeds added:", len(newPosSeeds)
        print "  total newNegSeeds added:", len(newNegSeeds)
        print ""
        i += 1
    #print what we found
    removeDuplicates(dictPosRestIDToFoodItemFoodItemVecSentSentVec)
    removeDuplicates(dictNegRestIDToFoodItemFoodItemVecSentSentVec)
    print "\n---Results---"
    restIDs = list(set(dictPosRestIDToFoodItemFoodItemVecSentSentVec.keys() + dictNegRestIDToFoodItemFoodItemVecSentSentVec.keys()))
    print "Restaurant count", len(restIDs)
    for restID in restIDs:
        print "Restaurant: ", restID
        if dictPosRestIDToFoodItemFoodItemVecSentSentVec.has_key(restID):
            print "    pos count", len(dictPosRestIDToFoodItemFoodItemVecSentSentVec.get(restID))
            #print "\n  positive items:"
            #for pos in dictPosRestIDToFoodItemFoodItemVecSentSentVec.get(restID):
                #pos is of the form [foodItem FoodItemVec Sent SentVecs, OriginalSent]
                #print "    ", pos[0], ":  ", pos[4]
        if dictNegRestIDToFoodItemFoodItemVecSentSentVec.has_key(restID):
            print "    neg count", len(dictNegRestIDToFoodItemFoodItemVecSentSentVec.get(restID))
            #print "\n  negative items:"
            #for neg in dictNegRestIDToFoodItemFoodItemVecSentSentVec.get(restID):
                #print "    ", neg[0], ":  ", neg[4]
        

        

    
    


        
        








