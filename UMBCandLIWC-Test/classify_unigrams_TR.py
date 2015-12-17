import os, csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics

#Go to the directory
os.chdir('C:\\Users\\user\\Documents\\CMPS242\\project')


#First Pass get just text for unigram extraction...
#Get the CSVs and generate traindata* and traintarget arrays
traindataQuote = []  
traindataResponse = []
traintarget = []

reader = csv.reader(open('development_set_all_instances.csv', 'r'))
for i, row in enumerate(reader):
    if i != 0:
        for j, item in enumerate(row):
            if j == 4: #annot_mturk_mean_response_agreement
                traintarget.append(float(item))
            if j == 90: #z_quote_text
                quote = item
            if j == 91: #z_response_text
                response = item
        traindataResponse.append(quote + response)

reader = csv.reader(open('test_set_all_instances.csv', 'r'))
for i, row in enumerate(reader):
    if i != 0:
        for j, item in enumerate(row):
            if j == 4: #annot_mturk_mean_response_agreement
                traintarget.append(float(item))
            if j == 90: #z_quote_text
                quote = item
            if j == 91: #z_response_text
                response = item
        traindataResponse.append(quote + response)
traintarget = np.array(traintarget)

#Read data to get unigrams...
tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer(binary=True)
X_traindataResponse_counts = count_vect.fit_transform(traindataResponse)
unigrams = X_traindataResponse_counts.toarray()

#Train and Test based on selected features (reserved test data)
#use devData and bestFeaturesIndex to create trainData, trainTarget
bestFeaturesIndex = list(range(0,25559)) #all unigrams + all features

trainDataFinal = unigrams[:5395,:]
trainTargetFinal = traintarget[:5395]

#use test_set_all_instances.csv and bestFeaturesIndex to create testData, testTarget
#First Pass get just text for unigram extraction...
print("working on 2nd unigrams..")
#Create unigram + features initial array
unigramsAndFeaturesTest = np.zeros((2847,25559)) 
unigramsAndFeaturesTest[:,:] = unigrams[5395:,:]

testData = unigramsAndFeaturesTest
testTarget = traintarget[5395:]

estimator = Pipeline([ ("imputer", Imputer()),
                    ("treeReg", DecisionTreeRegressor(max_depth=5)) ])
estimator.fit(trainDataFinal, trainTargetFinal)
predicted = estimator.predict(testData)
mseScore = mean_squared_error(testTarget, predicted)
print("mseScore: " + str(mseScore))


#F-scores calculations
#convert test scores to categorical values
testTargetCategorical = []
for val in testTarget:
    if val < -1:
        testTargetCategorical.append('disagree')
    elif val > 1:
        testTargetCategorical.append('agree')
    else:
        testTargetCategorical.append('neutral')
        
predictedCategorical = []
for val in predicted:
    if val < -1:
        predictedCategorical.append('disagree')
    elif val > 1:
        predictedCategorical.append('agree')
    else:
        predictedCategorical.append('neutral')
        
print(metrics.classification_report(testTargetCategorical, predictedCategorical))



