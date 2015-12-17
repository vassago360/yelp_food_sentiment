
getFoodandReviewVecs('111',"i like this big hotdogs. It tastes great! The burritos taste like shit", extFoodItems)

stopwords = set(nltk.corpus.stopwords.words('english'))

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for uid, line in enumerate(open(filename)):
            yield doc2vec.LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])

def sss(s1, s2):
    sss_url = "http://swoogle.umbc.edu/StsService/GetStsSim"
    try:
        response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2})
        return float(response.text.strip())
    except:
        print 'Error in getting similarity for %s: %s' % ((s1,s2), response)
        return 0.0
    
def createFoodItems():
    return []
    foodDictionary = open("foodTypes.gold.txt","r")
    foodItems = []
    for foodPair in foodDictionary:
        foodItem, foodType = foodPair.split("\t")
        foodItems.append(foodItem.lower())
    return foodItems

def getFoodandReviewVecs(review, foodItems, dictFoodItemVecToFoodItemName):

    review = """Five Stars Plus Plus!  \n\nWe are both native New Yorkers, 
    and have been on a quest for the last 28 years here in Arizona to find 
    the perfect New York pizza.  We've taste-tested all over town, 
    and we are so lucky to have found Gus's.  His pizza reminds us of home.  
    The ingredients are perfect - the crust is fantastic.  
    The restaurant is clean and the service and the staff are 
    friendly and inviting.  We are now loyal weekly patrons, 
    and are so happy to have him in our neighborhood."""
    
    getFoodandReviewVecs(review, foodItems, dictFoodItemVecToFoodItemName)    
        
    
    reviewFoodItems = []
    sents = nltk.sent_tokenize(review)
    for sidx in range(len(sents)):
        sent = sents[sidx]
        sent = ' '.join(sent.split()) # get rid of extra white spaces
        sent = sent.strip() # necessary? 
        if not sent: continue
        words = nltk.word_tokenize(sent)
        words_pos = nltk.pos_tag(words)
        for i in range(len(words_pos)):
            word, pos = words_pos[i] # TODO: use pos somewhere?
            word = word.strip()
            word.lower()
            if not word: continue
            if word in foodItems:
                print word, "is a food"
                reviewFoodItems.append(word)
    return foodItemVec, reviewSentVecs

"""usedDataBusIDs, dictUsedDataBusIDsToReview = get_data()
for busID in usedDataBusIDs:
    print "\n\n\n---------- ", busID, " ----------"
    for review in dictUsedDataBusIDsToReview.get(busID):
        print review
    print "---------------------------------------"  """
    
def getTopReviewSentencesID(reviewSentID, dictReviewSentIDToNbrsIndex, nbrsDistances, nbrsGraph, threshold=.9):
    #Given reviewSentence index value, find k nearest neighbors in nbrsGraph, get their distances in nbrsDistances
    # dictReviewSentenceIDToListofTopReviewSentenceID
    index = dictReviewSentIDToNbrsIndex.get(reviewSentID)
    nbrsDistances[index]


    s1 = "A small violin is being played by a girl"
    s2 = "a child is performing on a tiny instrument"
    s3 = "is"
    s4 = "is"
    sim = sss(s1, s2, type='relation', corpus='webbase')
    print sim
    
    
vec = getWordVec("girl")
print "word as a vector: ", vec.shape
vec = getDocVec("A small violin is being played by a girl")
print "doc as a vector: ", vec.shape


#MAKING temp dictReviewSencesIDToReviewSentence
#            dictReviewSentenceIDToReviewSentenceFeatures
#            dictReviewSentenceIDToListofTopReviewSentenceID


def getDocVec(doc):
    #model_path = os.path.join('..', 'word2vec-Test', 'trunk', 'GoogleNews-vectors-negative300.bin')
    model_path = os.path.join('..', 'word2vec-Test', 'trunk', 'vectors.bin')
    model = doc2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
    model.save(os.path.join('testsentences', 'saveModel'))
    model.load(os.path.join('testsentences', 'saveModel'))
    more_sentences = doc2vec.LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])
    #more_sentences = LabeledLineSentence(os.path.join('testsentences', 'testsentences.txt'))
    model.train(more_sentences)
    doc = doc.lower()
    if doc in model.vocab:
        vec = model.syn0norm[model.vocab[word].index]
    else:
        raise KeyError("word '%s' not in vocabulary" % word)
    return vec    

def knnToBeDone():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([[0,0]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    m = nbrs.kneighbors_graph(X)
    r, c, v = scipy.sparse.find(m)
    print distances
    index = dictReviewSentIDToNbrsIndex.get(reviewSentID)
    nbrsDistances[index]
    getTopReviewSentencesID(reviewSentID, dictReviewSentIDToNbrsIndex, nbrsDistances, nbrsGraph, threshold=.9)


def createVecsFromSentences(restIDsFoodItemsSents, wordWeights, vocab, sentences):
    dictRestIDToFoodItemToReviewSent = {}
    dictFoodItemToFoodItemVec = {}
    dictSentToSentVec = {}
    for restIDsFoodItemsSent in restIDsFoodItemsSents:
        restaurantID, foodItem, sentence = restIDsFoodItemsSent
        try: foodItemVec = getWordVec(foodItem) ; dictFoodItemToFoodItemVec[foodItem] = foodItemVec
        except: print "getWordVec(", foodItem, ") did not work.  Not including", sentence ; continue
        sentVec = getSentVec(sentence, sentences.index(sentence), wordWeights, vocab)
        if dictRestIDToFoodItemToReviewSent.has_key(restaurantID): #restaurant in dict already with at least one foodItem, Sentence
            foodItemsInThere = dictRestIDToFoodItemToReviewSent.get(restaurantID).keys()
            simFoodItem = getNN(foodItemsInThere, np.array([foodItemVec])) ####Use KNN to get what's super-similar
            if simFoodItem: #foodItem in dict already with at least one sentence
                dictRestIDToFoodItemToReviewSent.get(restaurantID).get(simFoodItem).append(sentence)
                dictSentToSentVec[sentence] = sentVec
            else: #new foodItem or one that's too far away from any other
                dictRestIDToFoodItemToReviewSent.get(restaurantID)[foodItem] = [sentence]
                dictFoodItemToFoodItemVec[foodItem] = foodItemVec
                dictSentToSentVec[sentence] = sentVec
        else: #new restaurant
            dictRestIDToFoodItemToReviewSent[restaurantID] = {foodItemVec:[sentence]}
            dictFoodItemToFoodItemVec[foodItem] = foodItemVec
            dictSentToSentVec[sentence] = sentVec
    return dictRestIDToFoodItemToReviewSent, dictFoodItemToFoodItemVec, dictSentToSentVec    




def preprocessSeeds(sents):
    for sidx in range(len(sents)):
        reviewFoodItems = []
        reconstructedSent = []
        sent = sents[sidx]
        sent = ' '.join(sent.split()) # get rid of extra white spaces
        sent = sent.strip() # necessary? 
        words = nltk.word_tokenize(sent)
        #words_pos = nltk.pos_tag(words)
        if not sent: continue
        for i, word in enumerate(words):
            word = word.strip()
            word = word.lower()
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
            restaurantIDFoodItemSentencelist.append([restaurantID, reviewFoodItem, ' '.join(reconstructedSent)])
    return restaurantIDFoodItemSentencelist



#do a pairwise comparison (no redundancies please) of foodItemVectors use similarity measure to get dictRestaurantIDToDictFoodItemVectorToReviewSentenceVectors
dictRestIDToFoodItemVecToReviewSentenceVectors = {}


#ONLINE TIME... Bootstrapping...
#during the bootstrapping: using the seeds [[restaurantID, foodItemVector]...] plugged into dictRestaurantIDToDictFoodItemVectorToReviewSentenceVectors generate dictRestaurantIDToDictReviewSentenceVectorToTopReviewSentenceVectors using sckit learn knn
dictRestaurantIDToDictReviewSentenceVectorToTopReviewSentenceVectors = {}

def itemsToVecs(items, dictToUse):
    vecs = []
    for item in items:
        vecs.append(dictToUse.get(item))
    return vecs

def getIthItemsFromDictToList(myDict, itemNum):
    items = []
    for key in myDict.keys():
        l = myDict.get(key)
        items.append(l[itemNum])
    return items

def getFoodItemsNN(foodItemsInThere, Y, withinDis=.1):
    X = itemsToVecs(foodItemsInThere, dictFoodItemToFoodItemVec)
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #Y = np.array([[0,0]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(Y)
    for i, dist in distances:
        if dis <= withinDis:
            return foodItemsInThere[indices[i]]
    return None








